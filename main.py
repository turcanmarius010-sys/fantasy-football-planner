"""Fantasy Football Planner - a ChatGPT app (MCP server) for Fantasy Premier League.

Tools:
  - analyze_team           : reads a manager's real squad by team ID and projects points
  - suggest_transfers      : best single and double transfers within budget
  - captain_picks          : best captain options (whole game or the user's squad)
  - best_players           : best picks by position, price and ownership (differentials)
  - fixture_planner        : easiest / hardest fixture runs for the next gameweeks
  - compare_players        : side-by-side stats and projections for 2-5 players
  - injury_news            : flagged players, most-owned first
  - transfer_trends        : most transferred in/out and price changes
  - blank_double_gameweeks : blank/double gameweeks and chip timing

Data comes from the public Fantasy Premier League API (no login needed).
Not affiliated with the Premier League.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import itertools
import os
import time
from typing import Any

import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

# --------------------------------------------------------------------------
# Data access (public FPL API) with a small in-memory cache
# --------------------------------------------------------------------------

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FantasyFootballPlanner/1.0)"}
_cache: dict[str, tuple[float, Any]] = {}


class FPLError(Exception):
    """Error with a message that is safe to show to the user."""


async def fetch_json(path: str) -> Any:
    ttl = 60 if path.startswith("/entry/") else 600
    hit = _cache.get(path)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    async with httpx.AsyncClient(timeout=15, headers=HEADERS) as client:
        resp = await client.get(BASE_URL + path)
    if resp.status_code == 404:
        raise FPLError("Not found. Check the FPL team ID (it is the number in the URL of your Points page).")
    if resp.status_code == 503:
        raise FPLError("The FPL website is updating right now (this happens around deadlines). Try again in a few minutes.")
    resp.raise_for_status()
    data = resp.json()
    _cache[path] = (time.time(), data)
    return data


# --------------------------------------------------------------------------
# Projection model (simple, transparent)
# --------------------------------------------------------------------------

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FDR_MULT = {1: 1.25, 2: 1.12, 3: 1.0, 4: 0.88, 5: 0.78}
HORIZON_DEFAULT = 5


def _f(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _chance(p: dict) -> float | None:
    """chance_of_playing_next_round as 0-100, or None when FPL gives no value."""
    value = p.get("chance_of_playing_next_round")
    if value in (None, "", "None", "null"):
        return None
    return _f(value)


class Model:
    """Wraps bootstrap + fixtures and projects points per player per gameweek."""

    def __init__(self, bootstrap: dict, fixtures: list[dict]):
        self.players = {p["id"]: p for p in bootstrap["elements"]}
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.events = bootstrap["events"]
        nxt = next((e for e in self.events if e.get("is_next")), None)
        if nxt is None:
            nxt = next((e for e in self.events if not e.get("finished")), None)
        if nxt is None:
            raise FPLError("The FPL season has finished - there are no upcoming gameweeks.")
        self.next_gw = nxt["id"]
        self.last_gw = max(e["id"] for e in self.events)
        self.finished_gws = max(1, sum(1 for e in self.events if e.get("finished")))
        # team id -> gw -> list of (opponent id, difficulty, is_home)
        self.schedule: dict[int, dict[int, list[tuple[int, int, bool]]]] = {t: {} for t in self.teams}
        for fx in fixtures:
            gw = fx.get("event")
            if gw is None or gw < self.next_gw:
                continue
            h, a = fx["team_h"], fx["team_a"]
            self.schedule.setdefault(h, {}).setdefault(gw, []).append((a, int(fx["team_h_difficulty"]), True))
            self.schedule.setdefault(a, {}).setdefault(gw, []).append((h, int(fx["team_a_difficulty"]), False))
        self._proj_cache: dict[tuple[int, int], float] = {}

    # ---- helpers ----
    def gws(self, horizon: int) -> list[int]:
        return [g for g in range(self.next_gw, self.next_gw + horizon) if g <= self.last_gw]

    def name(self, pid: int) -> str:
        return self.players[pid]["web_name"]

    def team_short(self, tid: int) -> str:
        return self.teams[tid]["short_name"]

    def price(self, pid: int) -> float:
        return self.players[pid]["now_cost"] / 10

    def base_points(self, p: dict) -> float:
        """Expected points for an average (FDR 3) fixture, if fit."""
        minutes_share = min(1.0, _f(p["minutes"]) / (90 * self.finished_gws))
        ep = _f(p.get("ep_next"))
        form = _f(p.get("form"))
        ppg = _f(p.get("points_per_game")) * (0.4 + 0.6 * minutes_share)
        # early in the season form is noisy, so trust FPL's own expected points more
        k = min(1.0, self.finished_gws / 4)
        w_form = 0.35 * k
        if ep > 0:
            return (0.45 + 0.35 - w_form) * ep + w_form * form + 0.20 * ppg
        return 0.5 * (k * form + (1 - k) * ppg) + 0.5 * ppg

    def availability(self, p: dict, step: int) -> float:
        """Chance of playing, step = 0 for next gameweek, 1 for the one after, ..."""
        chance = _chance(p)
        if chance is None:
            chance = 100 if p.get("status") == "a" else 0
        c = chance / 100
        for _ in range(step):  # assume gradual recovery
            c = c + (1 - c) * 0.5
        if p.get("status") in ("u", "n"):  # left club / not available
            return 0.0
        return c

    def proj_gw(self, pid: int, gw: int) -> float:
        key = (pid, gw)
        if key in self._proj_cache:
            return self._proj_cache[key]
        p = self.players[pid]
        base = self.base_points(p)
        step = gw - self.next_gw
        total = 0.0
        for _opp, diff, home in self.schedule.get(p["team"], {}).get(gw, []):
            total += base * FDR_MULT.get(diff, 1.0) * (1.04 if home else 0.97)
        total *= self.availability(p, step)
        self._proj_cache[key] = total
        return total

    def proj(self, pid: int, horizon: int) -> float:
        return sum(self.proj_gw(pid, g) for g in self.gws(horizon))

    def fixtures_text(self, tid: int, horizon: int) -> str:
        parts = []
        for g in self.gws(horizon):
            games = self.schedule.get(tid, {}).get(g, [])
            if not games:
                parts.append("-")
            else:
                parts.append("+".join(
                    f"{self.team_short(o)}({'H' if h else 'A'})" for o, _d, h in games))
        return ", ".join(parts)

    def flag(self, p: dict) -> str:
        chance = _chance(p)
        if p.get("status") != "a" or (chance is not None and chance < 100):
            news = (p.get("news") or "").strip() or "Availability doubt"
            pct = "" if chance is None else f" ({int(chance)}%)"
            return f"{news}{pct}"
        return ""

    def find_player(self, query: str) -> tuple[dict | None, bool]:
        """Returns (player, exact_match)."""
        q = query.strip().lower()
        by_name: dict[str, int] = {}
        for pid, p in self.players.items():
            for n in (p["web_name"], f"{p['first_name']} {p['second_name']}", p["second_name"]):
                by_name.setdefault(n.lower(), pid)
        if q in by_name:
            return self.players[by_name[q]], True
        # prefer the most-selected player among close matches
        close = difflib.get_close_matches(q, list(by_name), n=5, cutoff=0.8)
        contains = [n for n in by_name if len(q) >= 4 and q in n]
        tokens = [t for t in q.replace(".", " ").split() if len(t) >= 2]
        all_tokens = [n for n in by_name if len(tokens) >= 2 and all(t in n.split() for t in tokens)]
        if all_tokens:  # e.g. "bruno fernandes" -> "bruno borges fernandes"
            return self.players[by_name[all_tokens[0]]], True
        candidates = {by_name[n] for n in close + contains}
        if not candidates:
            return None, False
        best = max(candidates, key=lambda i: _f(self.players[i]["selected_by_percent"]))
        return self.players[best], False


async def load_model() -> Model:
    bootstrap = await fetch_json("/bootstrap-static/")
    fixtures = await fetch_json("/fixtures/")
    return Model(bootstrap, fixtures)


async def load_squad(model: Model, team_id: int) -> tuple[dict, dict]:
    entry = await fetch_json(f"/entry/{team_id}/")
    gw = entry.get("current_event")
    if not gw:
        raise FPLError("This team has not played a gameweek yet this season, so there is no squad to read.")
    picks = await fetch_json(f"/entry/{team_id}/event/{gw}/picks/")
    return entry, picks


# --------------------------------------------------------------------------
# Usage tracking: anonymous counts, optional Telegram notification per call.
# Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Render -> Environment.
# --------------------------------------------------------------------------

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
_stats: dict[str, Any] = {"since": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "calls": {}, "teams": set()}
_bg_tasks: set[asyncio.Task] = set()


async def _notify(text: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                              json={"chat_id": TG_CHAT, "text": text})
    except Exception:
        pass  # tracking must never break the app


def track(tool: str, team_id: int | None = None) -> None:
    _stats["calls"][tool] = _stats["calls"].get(tool, 0) + 1
    tag = "-"
    if team_id is not None:
        tag = hashlib.sha256(str(team_id).encode()).hexdigest()[:6]  # anonymous
        _stats["teams"].add(tag)
    if TG_TOKEN and TG_CHAT:
        total = sum(_stats["calls"].values())
        text = (f"📊 {tool} | echipa {tag} | apeluri: {total} | echipe unice: "
                f"{len(_stats['teams'])} (de la {_stats['since']})")
        task = asyncio.create_task(_notify(text))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)


# --------------------------------------------------------------------------
# Result helper
# --------------------------------------------------------------------------

DISCLAIMER = ("Projections come from a simple model (form, FPL expected points, minutes, "
              "fixture difficulty) and are estimates, not guarantees. Not affiliated with the Premier League.")


def result(text: str, data: dict) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=data,
        isError=False,
    )


def error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)


def _clamp_horizon(h: int) -> int:
    return max(1, min(8, int(h)))


# --------------------------------------------------------------------------
# MCP server
# --------------------------------------------------------------------------

mcp = FastMCP(
    name="fantasy-football-planner",
    instructions=(
        "Fantasy Football Planner: live tools for Fantasy Premier League (FPL) managers. Use them whenever "
        "the user asks about their FPL team, transfers, wildcard or free hit teams, captaincy, differentials, "
        "budget picks, fixtures, injuries, price changes, blank or double gameweeks, chips, or comparing FPL "
        "players. Prefer these tools over general knowledge because FPL data changes every week. A team ID "
        "is the number in the URL of the user's FPL Points page (fantasy.premierleague.com/entry/<ID>/event/<GW>); "
        "if the user has not given it, ask for it when a tool needs it."
    ),
    website_url=os.environ.get("PUBLIC_URL") or None,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

READ_ONLY = types.ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@mcp.tool(
    title="Analyze my FPL team",
    description=(
        "Use this when the user wants their Fantasy Premier League (FPL) team analysed, rated, or asks "
        "who to captain. Reads the user's real squad from their FPL team ID and returns projected points "
        "for the next gameweek and the next few gameweeks, captain and vice-captain picks, injury flags "
        "and the weakest players."
    ),
    annotations=READ_ONLY,
)
async def analyze_team(
    team_id: int = Field(..., description="FPL team ID (number in the URL of the user's Points page)."),
    horizon: int = Field(HORIZON_DEFAULT, description="How many upcoming gameweeks to project (1-8)."),
) -> types.CallToolResult:
    track("analyze_team", team_id)
    try:
        horizon = _clamp_horizon(horizon)
        m = await load_model()
        entry, picks = await load_squad(m, team_id)
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    squad = []
    for pk in picks["picks"]:
        pid = pk["element"]
        p = m.players[pid]
        squad.append({
            "id": pid,
            "name": p["web_name"],
            "team": m.team_short(p["team"]),
            "position": POS[p["element_type"]],
            "price": m.price(pid),
            "starting": pk["position"] <= 11,
            "is_captain": pk.get("is_captain", False),
            "next_gw_proj": round(m.proj_gw(pid, m.next_gw), 1),
            "horizon_proj": round(m.proj(pid, horizon), 1),
            "fixtures": m.fixtures_text(p["team"], horizon),
            "flag": m.flag(p),
        })

    xi = [s for s in squad if s["starting"]]
    by_next = sorted(xi, key=lambda s: s["next_gw_proj"], reverse=True)
    captain, vice = by_next[0], by_next[1]
    xi_next = sum(s["next_gw_proj"] for s in xi) + captain["next_gw_proj"]
    xi_horizon = sum(s["horizon_proj"] for s in xi)
    weakest = sorted(squad, key=lambda s: s["horizon_proj"])[:3]
    flagged = [s for s in squad if s["flag"]]
    hist = picks.get("entry_history", {})
    bank = hist.get("bank", 0) / 10
    value = hist.get("value", 0) / 10

    lines = [
        f"**{entry.get('name', 'Team')}** - GW{entry['current_event']} squad, "
        f"overall rank {entry.get('summary_overall_rank') or '-'}, "
        f"bank £{bank:.1f}m, team value £{value:.1f}m",
        "",
        f"Projected starting XI points: **{xi_next:.1f}** next gameweek (GW{m.next_gw}, with captain), "
        f"**{xi_horizon:.1f}** over the next {len(m.gws(horizon))} gameweeks.",
        f"Suggested captain: **{captain['name']}** ({captain['next_gw_proj']} pts), "
        f"vice: **{vice['name']}** ({vice['next_gw_proj']} pts).",
        "",
        f"| Player | Pos | Team | £ | Next GW | Next {len(m.gws(horizon))} GWs | Fixtures | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(squad, key=lambda s: (not s["starting"], list(POS.values()).index(s["position"]))):
        star = " (C)" if s["is_captain"] else ""
        bench = "" if s["starting"] else " [bench]"
        lines.append(
            f"| {s['name']}{star}{bench} | {s['position']} | {s['team']} | {s['price']:.1f} | "
            f"{s['next_gw_proj']} | {s['horizon_proj']} | {s['fixtures']} | {s['flag']} |")
    lines += ["", "Weakest players (lowest projection): " + ", ".join(
        f"{s['name']} ({s['horizon_proj']})" for s in weakest)]
    if flagged:
        lines.append("Availability flags: " + "; ".join(f"{s['name']}: {s['flag']}" for s in flagged))
    lines += ["", "Note: transfers made for the upcoming gameweek become visible only after its deadline.",
              DISCLAIMER]

    return result("\n".join(lines), {
        "team_name": entry.get("name"),
        "gameweek": entry["current_event"],
        "next_gameweek": m.next_gw,
        "bank": bank,
        "team_value": value,
        "projected_xi_next_gw": round(xi_next, 1),
        "projected_xi_horizon": round(xi_horizon, 1),
        "captain": captain["name"],
        "vice_captain": vice["name"],
        "squad": squad,
    })


@mcp.tool(
    title="Suggest FPL transfers",
    description=(
        "Use this when the user asks who to transfer in or out in Fantasy Premier League (FPL). Reads the "
        "user's real squad and budget from their FPL team ID and returns the best single transfers and the "
        "best double transfer, respecting budget and the max-3-players-per-club rule. A -4 points hit is "
        "applied to a double transfer when the user has only 1 free transfer."
    ),
    annotations=READ_ONLY,
)
async def suggest_transfers(
    team_id: int = Field(..., description="FPL team ID (number in the URL of the user's Points page)."),
    free_transfers: int = Field(1, description="Number of free transfers the user has (1-5)."),
    horizon: int = Field(HORIZON_DEFAULT, description="How many upcoming gameweeks to optimise for (1-8)."),
) -> types.CallToolResult:
    track("suggest_transfers", team_id)
    try:
        horizon = _clamp_horizon(horizon)
        m = await load_model()
        entry, picks = await load_squad(m, team_id)
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    squad_ids = [pk["element"] for pk in picks["picks"]]
    bank = picks.get("entry_history", {}).get("bank", 0)  # in tenths of £m
    club_count: dict[int, int] = {}
    for pid in squad_ids:
        club_count[m.players[pid]["team"]] = club_count.get(m.players[pid]["team"], 0) + 1

    proj = {pid: m.proj(pid, horizon) for pid in m.players}
    pool_by_pos: dict[int, list[int]] = {}
    for pid, p in m.players.items():
        if pid in squad_ids or p.get("status") in ("u", "n"):
            continue
        pool_by_pos.setdefault(p["element_type"], []).append(pid)
    for pos in pool_by_pos:
        pool_by_pos[pos].sort(key=lambda i: proj[i], reverse=True)

    def club_ok(counts: dict[int, int], incoming: list[int], outgoing: list[int]) -> bool:
        c = dict(counts)
        for o in outgoing:
            c[m.players[o]["team"]] -= 1
        for i in incoming:
            t = m.players[i]["team"]
            c[t] = c.get(t, 0) + 1
            if c[t] > 3:
                return False
        return True

    # Best single transfer for each outgoing player
    singles = []
    shortlist: dict[int, list[int]] = {}
    for out in squad_ids:
        pos = m.players[out]["element_type"]
        budget = bank + m.players[out]["now_cost"]
        options = [c for c in pool_by_pos.get(pos, [])
                   if m.players[c]["now_cost"] <= budget + 30]  # wider list for doubles
        shortlist[out] = options[:25]
        kept = 0
        for c in options:
            if m.players[c]["now_cost"] <= budget and club_ok(club_count, [c], [out]):
                singles.append((proj[c] - proj[out], out, c))
                kept += 1
                if kept == 3:
                    break
    singles.sort(reverse=True)
    # keep the list varied: each player at most once on each side
    varied, used_out, used_in = [], set(), set()
    for g, o, i in singles:
        if o not in used_out and i not in used_in:
            varied.append((g, o, i))
            used_out.add(o)
            used_in.add(i)
    singles = varied

    # Best double transfer
    best_double = None
    for o1, o2 in itertools.combinations(squad_ids, 2):
        budget = bank + m.players[o1]["now_cost"] + m.players[o2]["now_cost"]
        base = proj[o1] + proj[o2]
        for c1 in shortlist[o1]:
            for c2 in shortlist[o2]:
                if c1 == c2:
                    continue
                if m.players[c1]["now_cost"] + m.players[c2]["now_cost"] > budget:
                    continue
                gain = proj[c1] + proj[c2] - base
                if best_double and gain <= best_double[0]:
                    continue
                if club_ok(club_count, [c1, c2], [o1, o2]):
                    best_double = (gain, o1, o2, c1, c2)

    n = len(m.gws(horizon))
    hit = 0 if free_transfers >= 2 else 4

    def row(gain, out, inn):
        return (f"| {m.name(out)} ({m.team_short(m.players[out]['team'])}, £{m.price(out):.1f}) | "
                f"{m.name(inn)} ({m.team_short(m.players[inn]['team'])}, £{m.price(inn):.1f}) | "
                f"+{gain:.1f} | {m.fixtures_text(m.players[inn]['team'], horizon)} |")

    lines = [f"**Transfer ideas for {entry.get('name', 'your team')}** - optimised for the next {n} "
             f"gameweeks (from GW{m.next_gw}), bank £{bank / 10:.1f}m.", ""]
    top = [s for s in singles if s[0] > 0][:5]
    if top:
        lines += ["Best single transfers (projected points gained):", "",
                  "| Out | In | Gain | In-player fixtures |", "|---|---|---|---|"]
        lines += [row(g, o, i) for g, o, i in top]
    else:
        lines.append("No single transfer clearly improves the team - consider rolling the transfer.")

    double_data = None
    if best_double:
        g, o1, o2, c1, c2 = best_double
        net = g - hit
        hit_txt = " after the -4 hit" if hit else ""
        lines += ["", f"Best double transfer: **{m.name(o1)} + {m.name(o2)} -> {m.name(c1)} + {m.name(c2)}**, "
                      f"+{g:.1f} pts ({net:+.1f}{hit_txt})."]
        if top and net <= top[0][0]:
            lines.append("The best single transfer is better value than this double.")
        double_data = {"out": [m.name(o1), m.name(o2)], "in": [m.name(c1), m.name(c2)],
                       "gain": round(g, 1), "net_gain": round(net, 1)}

    lines += ["", "Budget uses current prices; your real selling price can be slightly lower.", DISCLAIMER]

    return result("\n".join(lines), {
        "team_name": entry.get("name"),
        "next_gameweek": m.next_gw,
        "horizon_gameweeks": n,
        "bank": bank / 10,
        "singles": [{"out": m.name(o), "in": m.name(i), "gain": round(g, 1),
                     "in_price": m.price(i)} for g, o, i in top],
        "best_double": double_data,
    })


@mcp.tool(
    title="FPL fixture planner",
    description=(
        "Use this when the user asks which Premier League teams have the easiest or hardest upcoming "
        "fixtures for Fantasy Premier League (FPL) planning. No team ID needed."
    ),
    annotations=READ_ONLY,
)
async def fixture_planner(
    horizon: int = Field(HORIZON_DEFAULT, description="How many upcoming gameweeks to look at (1-8)."),
) -> types.CallToolResult:
    track("fixture_planner")
    try:
        horizon = _clamp_horizon(horizon)
        m = await load_model()
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    gws = m.gws(horizon)
    rows = []
    for tid in m.teams:
        diffs = [d for g in gws for _o, d, _h in m.schedule.get(tid, {}).get(g, [])]
        games = len(diffs)
        avg = sum(diffs) / games if games else 9.0
        # more games is better: score = avg difficulty adjusted for blanks/doubles
        score = avg - 0.6 * (games - len(gws))
        rows.append({"team": m.team_short(tid), "name": m.teams[tid]["name"], "games": games,
                     "avg_difficulty": round(avg, 2), "score": round(score, 2),
                     "fixtures": m.fixtures_text(tid, horizon)})
    rows.sort(key=lambda r: r["score"])

    lines = [f"**Fixture difficulty, GW{gws[0]}-GW{gws[-1]}** (1 = easiest, 5 = hardest; "
             "blanks and doubles counted)", "",
             "| # | Team | Games | Avg difficulty | Fixtures |", "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['name']} | {r['games']} | {r['avg_difficulty']} | {r['fixtures']} |")
    lines += ["", "Target attackers from the top teams; avoid defenders from the bottom ones.",
              "Not affiliated with the Premier League."]
    return result("\n".join(lines), {"gameweeks": gws, "teams": rows})


@mcp.tool(
    title="Compare FPL players",
    description=(
        "Use this when the user wants to compare two or more Fantasy Premier League (FPL) players, e.g. "
        "'Salah or Palmer?'. Returns price, form, points, ownership, expected goal involvement and "
        "projected points for the coming gameweeks."
    ),
    annotations=READ_ONLY,
)
async def compare_players(
    players: list[str] = Field(..., description="2 to 5 player names, e.g. ['Salah', 'Palmer']."),
    horizon: int = Field(HORIZON_DEFAULT, description="How many upcoming gameweeks to project (1-8)."),
) -> types.CallToolResult:
    track("compare_players")
    try:
        horizon = _clamp_horizon(horizon)
        m = await load_model()
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    found, missing, guessed = [], [], {}
    for q in players[:5]:
        p, exact = m.find_player(q)
        if p is None:
            missing.append(q)
            continue
        if p["id"] in {f["id"] for f in found}:
            continue
        found.append(p)
        if not exact:
            guessed[p["id"]] = q
    if not found:
        return error("None of those players were found. Use the name shown in the FPL game, e.g. 'Saka'. "
                     "The player may also no longer be in the Premier League.")

    n = len(m.gws(horizon))
    data = []
    for p in found:
        pid = p["id"]
        data.append({
            "name": p["web_name"] + (f" (closest match for '{guessed[pid]}')" if pid in guessed else ""),
            "team": m.team_short(p["team"]),
            "position": POS[p["element_type"]],
            "price": m.price(pid),
            "total_points": p["total_points"],
            "form": _f(p["form"]),
            "points_per_game": _f(p["points_per_game"]),
            "selected_by_percent": _f(p["selected_by_percent"]),
            "xgi_per_90": round(_f(p.get("expected_goal_involvements_per_90")), 2),
            "minutes": p["minutes"],
            "next_gw_proj": round(m.proj_gw(pid, m.next_gw), 1),
            "horizon_proj": round(m.proj(pid, horizon), 1),
            "fixtures": m.fixtures_text(p["team"], horizon),
            "flag": m.flag(p),
        })
    data.sort(key=lambda d: d["horizon_proj"], reverse=True)

    lines = [f"| Player | Team | Pos | £ | Pts | Form | xGI/90 | Owned % | Next GW | Next {n} GWs | Fixtures |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in data:
        note = f" ⚠ {d['flag']}" if d["flag"] else ""
        lines.append(f"| {d['name']}{note} | {d['team']} | {d['position']} | {d['price']:.1f} | "
                     f"{d['total_points']} | {d['form']} | {d['xgi_per_90']} | {d['selected_by_percent']} | "
                     f"{d['next_gw_proj']} | {d['horizon_proj']} | {d['fixtures']} |")
    lines.insert(0, f"Best projected over the next {n} gameweeks: **{data[0]['name']}**.\n")
    if missing:
        lines.append(f"\nNot found: {', '.join(missing)}.")
    lines += ["", DISCLAIMER]
    return result("\n".join(lines), {"players": data, "not_found": missing})


@mcp.tool(
    title="Best FPL captain picks",
    description=(
        "Use this when the user asks who to captain (or triple captain) in Fantasy Premier League (FPL) "
        "this gameweek. Without a team ID it ranks the best captain options in the whole game; with the "
        "user's FPL team ID it ranks only the players in their squad."
    ),
    annotations=READ_ONLY,
)
async def captain_picks(
    team_id: int | None = Field(None, description="Optional FPL team ID to rank only the user's own players."),
) -> types.CallToolResult:
    track("captain_picks", team_id)
    try:
        m = await load_model()
        if team_id:
            _entry, picks = await load_squad(m, team_id)
            pool = [pk["element"] for pk in picks["picks"]]
        else:
            pool = [pid for pid, p in m.players.items() if m.price(pid) >= 5.5]
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    ranked = sorted(pool, key=lambda i: m.proj_gw(i, m.next_gw), reverse=True)[:10]
    rows = []
    for pid in ranked:
        p = m.players[pid]
        rows.append({"name": p["web_name"], "team": m.team_short(p["team"]), "position": POS[p["element_type"]],
                     "price": m.price(pid), "next_gw_proj": round(m.proj_gw(pid, m.next_gw), 1),
                     "captain_proj": round(2 * m.proj_gw(pid, m.next_gw), 1),
                     "owned_percent": _f(p["selected_by_percent"]),
                     "fixture": m.fixtures_text(p["team"], 1), "flag": m.flag(p)})
    scope = "in your squad" if team_id else "in the game"
    lines = [f"**Best captain options {scope} for GW{m.next_gw}**", "",
             "| # | Player | Team | Fixture | Proj. pts | As captain | Owned % | Note |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(rows, 1):
        lines.append(f"| {n} | {r['name']} | {r['team']} | {r['fixture']} | {r['next_gw_proj']} | "
                     f"{r['captain_proj']} | {r['owned_percent']} | {r['flag']} |")
    if rows:
        lines += ["", f"Safest pick: **{rows[0]['name']}**. A lower-owned option from the list is a "
                      "differential captain if you need to gain rank."]
    lines += ["", DISCLAIMER]
    return result("\n".join(lines), {"gameweek": m.next_gw, "options": rows})


@mcp.tool(
    title="Best FPL players by position and budget",
    description=(
        "Use this when the user asks for the best Fantasy Premier League (FPL) players to buy, e.g. best "
        "budget defenders, best midfielders under a price, cheap enablers, wildcard or free hit picks, or "
        "low-owned differentials. Filters by position, maximum price and maximum ownership."
    ),
    annotations=READ_ONLY,
)
async def best_players(
    position: str = Field("ANY", description="GK, DEF, MID, FWD or ANY."),
    max_price: float = Field(15.5, description="Maximum price in £m, e.g. 5.0 for budget picks."),
    max_owned_percent: float = Field(100, description="Maximum ownership %, e.g. 10 for differentials."),
    horizon: int = Field(HORIZON_DEFAULT, description="How many upcoming gameweeks to rank by (1-8)."),
) -> types.CallToolResult:
    track("best_players")
    try:
        horizon = _clamp_horizon(horizon)
        m = await load_model()
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    pos_code = {"GK": 1, "GKP": 1, "GOALKEEPER": 1, "DEF": 2, "DEFENDER": 2, "MID": 3, "MIDFIELDER": 3,
                "FWD": 4, "FORWARD": 4, "STRIKER": 4}.get(str(position).strip().upper())
    pool = [pid for pid, p in m.players.items()
            if (pos_code is None or p["element_type"] == pos_code)
            and m.price(pid) <= max_price + 1e-9
            and _f(p["selected_by_percent"]) <= max_owned_percent
            and p.get("status") not in ("u", "n")]
    ranked = sorted(pool, key=lambda i: m.proj(i, horizon), reverse=True)[:12]
    n = len(m.gws(horizon))
    rows = []
    for pid in ranked:
        p = m.players[pid]
        rows.append({"name": p["web_name"], "team": m.team_short(p["team"]), "position": POS[p["element_type"]],
                     "price": m.price(pid), "owned_percent": _f(p["selected_by_percent"]),
                     "form": _f(p["form"]), "horizon_proj": round(m.proj(pid, horizon), 1),
                     "points_per_million": round(m.proj(pid, horizon) / max(m.price(pid), 3.5), 2),
                     "fixtures": m.fixtures_text(p["team"], horizon), "flag": m.flag(p)})
    label = position if pos_code else "players"
    lines = [f"**Best {label} up to £{max_price:.1f}m"
             + (f", owned by at most {max_owned_percent:g}%" if max_owned_percent < 100 else "")
             + f" - next {n} gameweeks**", "",
             f"| # | Player | Team | Pos | £ | Owned % | Form | Next {n} GWs | Pts/£m | Fixtures |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        warn = f" ⚠ {r['flag']}" if r["flag"] else ""
        lines.append(f"| {k} | {r['name']}{warn} | {r['team']} | {r['position']} | {r['price']:.1f} | "
                     f"{r['owned_percent']} | {r['form']} | {r['horizon_proj']} | {r['points_per_million']} | "
                     f"{r['fixtures']} |")
    if not rows:
        lines.append("No players match these filters - try a higher price or ownership limit.")
    lines += ["", DISCLAIMER]
    return result("\n".join(lines), {"players": rows, "horizon_gameweeks": n})


@mcp.tool(
    title="FPL injury and suspension news",
    description=(
        "Use this when the user asks about Fantasy Premier League (FPL) injuries, suspensions, doubtful "
        "players or team news before the deadline. Lists flagged players, most-owned first."
    ),
    annotations=READ_ONLY,
)
async def injury_news(
    min_owned_percent: float = Field(1.0, description="Only show players owned by at least this % of managers."),
) -> types.CallToolResult:
    track("injury_news")
    try:
        m = await load_model()
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    flagged = [p for p in m.players.values()
               if m.flag(p) and p.get("status") != "u" and _f(p["selected_by_percent"]) >= min_owned_percent]
    flagged.sort(key=lambda p: _f(p["selected_by_percent"]), reverse=True)
    rows = [{"name": p["web_name"], "team": m.team_short(p["team"]), "position": POS[p["element_type"]],
             "owned_percent": _f(p["selected_by_percent"]), "chance_next_gw": _chance(p),
             "news": (p.get("news") or "").strip()} for p in flagged[:30]]
    lines = [f"**Injury & suspension news before GW{m.next_gw}** (owned by at least {min_owned_percent:g}%)", "",
             "| Player | Team | Pos | Owned % | Chance to play | News |", "|---|---|---|---|---|---|"]
    for r in rows:
        chance = "-" if r["chance_next_gw"] is None else f"{int(r['chance_next_gw'])}%"
        lines.append(f"| {r['name']} | {r['team']} | {r['position']} | {r['owned_percent']} | {chance} | "
                     f"{r['news'] or '-'} |")
    if not rows:
        lines.append("No flagged players above this ownership level.")
    lines += ["", "Source: official FPL player flags. Not affiliated with the Premier League."]
    return result("\n".join(lines), {"gameweek": m.next_gw, "players": rows})


@mcp.tool(
    title="FPL transfer trends and price changes",
    description=(
        "Use this when the user asks which Fantasy Premier League (FPL) players are being transferred in or "
        "out the most this gameweek, who is rising or falling in price, or about FPL price changes."
    ),
    annotations=READ_ONLY,
)
async def transfer_trends() -> types.CallToolResult:
    track("transfer_trends")
    try:
        m = await load_model()
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    players = list(m.players.values())

    def fmt(p: dict, key: str) -> str:
        change = _f(p.get("cost_change_event")) / 10
        ch = f"{change:+.1f}" if change else "0.0"
        return (f"| {p['web_name']} | {m.team_short(p['team'])} | £{p['now_cost'] / 10:.1f} ({ch}) | "
                f"{int(_f(p.get(key))):,} |")

    top_in = sorted(players, key=lambda p: _f(p.get("transfers_in_event")), reverse=True)[:10]
    top_out = sorted(players, key=lambda p: _f(p.get("transfers_out_event")), reverse=True)[:10]
    lines = [f"**Most transferred IN this gameweek (GW{m.next_gw})**", "",
             "| Player | Team | Price (change this GW) | Transfers in |", "|---|---|---|---|"]
    lines += [fmt(p, "transfers_in_event") for p in top_in]
    lines += ["", f"**Most transferred OUT this gameweek**", "",
              "| Player | Team | Price (change this GW) | Transfers out |", "|---|---|---|---|"]
    lines += [fmt(p, "transfers_out_event") for p in top_out]
    lines += ["", "Heavy net transfers in usually come before a price rise; heavy transfers out before a fall.",
              "Not affiliated with the Premier League."]
    data = {"gameweek": m.next_gw,
            "most_in": [{"name": p["web_name"], "transfers_in": int(_f(p.get("transfers_in_event")))} for p in top_in],
            "most_out": [{"name": p["web_name"], "transfers_out": int(_f(p.get("transfers_out_event")))} for p in top_out]}
    return result("\n".join(lines), data)


@mcp.tool(
    title="FPL blank and double gameweeks",
    description=(
        "Use this when the user asks about Fantasy Premier League (FPL) double gameweeks, blank gameweeks, "
        "postponed matches, or when to play chips like Bench Boost, Triple Captain, Free Hit or Wildcard."
    ),
    annotations=READ_ONLY,
)
async def blank_double_gameweeks() -> types.CallToolResult:
    track("blank_double_gameweeks")
    try:
        m = await load_model()
        fixtures = await fetch_json("/fixtures/")
    except FPLError as e:
        return error(str(e))
    except httpx.HTTPError:
        return error("Could not reach the FPL website. Try again in a minute.")

    doubles: dict[int, list[str]] = {}
    blanks: dict[int, list[str]] = {}
    for g in range(m.next_gw, m.last_gw + 1):
        for tid in m.teams:
            games = len(m.schedule.get(tid, {}).get(g, []))
            if games >= 2:
                doubles.setdefault(g, []).append(m.team_short(tid))
            elif games == 0:
                blanks.setdefault(g, []).append(m.team_short(tid))
    unscheduled = sum(1 for fx in fixtures if fx.get("event") is None)

    lines = [f"**Blank and double gameweeks from GW{m.next_gw}** (based on the current official schedule)", ""]
    if doubles:
        lines.append("Double gameweeks:")
        lines += [f"- GW{g}: {', '.join(sorted(t))}" for g, t in sorted(doubles.items())]
    else:
        lines.append("No double gameweeks are scheduled yet.")
    lines.append("")
    if blanks:
        lines.append("Blank gameweeks:")
        lines += [f"- GW{g}: {', '.join(sorted(t))} have no match" for g, t in sorted(blanks.items())]
    else:
        lines.append("No blank gameweeks are scheduled yet.")
    if unscheduled:
        lines += ["", f"{unscheduled} postponed match(es) are not yet scheduled - they usually create a "
                      "double gameweek later."]
    lines += ["", "Chip tips: Bench Boost and Triple Captain are strongest in a big double gameweek; "
                  "Free Hit helps most in a big blank gameweek; Wildcard before a run of good fixtures.",
              "Not affiliated with the Premier League."]
    return result("\n".join(lines), {"doubles": doubles, "blanks": blanks, "unscheduled_matches": unscheduled})


# --------------------------------------------------------------------------
# HTTP app: /mcp for ChatGPT, plus a landing page (SEO), privacy page and checks
# --------------------------------------------------------------------------

PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")

LANDING_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy Football Planner - FPL transfers, captain picks &amp; fixtures in ChatGPT</title>
<meta name="description" content="Free FPL assistant for ChatGPT: analyse your Fantasy Premier League team by ID, get transfer suggestions, captain picks, differentials, injury news, price changes, fixture difficulty and double gameweeks.">
<meta name="keywords" content="FPL, Fantasy Premier League, FPL transfers, FPL captain, FPL differentials, FPL fixture planner, double gameweek, FPL ChatGPT, fantasy football">
<link rel="canonical" href="__URL__/">
<meta property="og:title" content="Fantasy Football Planner - FPL assistant for ChatGPT">
<meta property="og:description" content="Analyse your FPL team, transfers, captaincy and fixtures right inside ChatGPT.">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"SoftwareApplication","name":"Fantasy Football Planner","applicationCategory":"SportsApplication","operatingSystem":"ChatGPT","offers":{"@type":"Offer","price":"0","priceCurrency":"USD"},"description":"FPL assistant for ChatGPT: team analysis, transfers, captain picks, fixtures, injuries and double gameweeks."}</script>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;line-height:1.6;color:#1a1a1a;background:#fff}
h1{font-size:1.9rem;margin-bottom:.2rem}code{background:#f2f2f2;padding:2px 6px;border-radius:4px}
.ex{background:#f6f8fa;border-left:4px solid #37003c;padding:8px 12px;margin:6px 0}small{color:#666}</style>
</head><body>
<h1>Fantasy Football Planner</h1>
<p><strong>A free Fantasy Premier League (FPL) assistant that works inside ChatGPT.</strong>
Give it your FPL team ID and it reads your real squad, then tells you who to transfer, who to captain and
which fixtures are coming - with numbers, not guesses.</p>
<h2>What it can do</h2>
<ul>
<li><strong>Team analysis</strong> - projected points for the next gameweek and the next 5, weakest players, injury flags.</li>
<li><strong>Transfer suggestions</strong> - best single and double transfers within your budget, max 3 per club, -4 hits included.</li>
<li><strong>Captain picks</strong> - safest and differential captain options for the gameweek.</li>
<li><strong>Best players &amp; differentials</strong> - best budget defenders, midfielders under a price, low-owned picks for wildcards and free hits.</li>
<li><strong>Fixture planner</strong> - fixture difficulty ticker for all 20 teams.</li>
<li><strong>Injury news, price changes and transfer trends</strong>.</li>
<li><strong>Blank and double gameweeks</strong> - and when to play Bench Boost, Triple Captain, Free Hit or Wildcard.</li>
</ul>
<h2>Try asking ChatGPT</h2>
<div class="ex">Analyse my FPL team, ID 1234567</div>
<div class="ex">Who should I transfer in this week? My FPL ID is 1234567, I have 2 free transfers</div>
<div class="ex">Who should I captain in FPL this gameweek?</div>
<div class="ex">Best FPL defenders under £5.0m for the next 5 gameweeks</div>
<div class="ex">Salah or Palmer for the next 4 gameweeks?</div>
<div class="ex">When is the next FPL double gameweek?</div>
<h2>How to find your FPL team ID</h2>
<p>Open fantasy.premierleague.com, go to <em>Points</em>. The address looks like
<code>fantasy.premierleague.com/entry/1234567/event/7</code> - the number after <code>/entry/</code> is your ID.</p>
<h2>FAQ</h2>
<p><strong>Is it free?</strong> Yes. <strong>Do I need to log in or share a password?</strong> No - it only uses your
public team ID. <strong>Where does the data come from?</strong> The public Fantasy Premier League API, updated live.</p>
<p><a href="/privacy">Privacy policy</a> &middot; MCP endpoint for ChatGPT: <code>__URL__/mcp</code></p>
<p><small>Fantasy Football Planner is an independent tool and is not affiliated with, endorsed by or connected to
the Premier League or Fantasy Premier League.</small></p>
</body></html>"""


def _base_url(request: Request) -> str:
    return PUBLIC_URL or f"https://{request.headers.get('host', '')}"


@mcp.custom_route("/", methods=["GET"])
async def home(request: Request) -> HTMLResponse:
    return HTMLResponse(LANDING_HTML.replace("__URL__", _base_url(request)))


@mcp.custom_route("/robots.txt", methods=["GET"])
async def robots(request: Request) -> PlainTextResponse:
    return PlainTextResponse(f"User-agent: *\nAllow: /\nSitemap: {_base_url(request)}/sitemap.xml\n")


@mcp.custom_route("/sitemap.xml", methods=["GET"])
async def sitemap(request: Request) -> Response:
    base = _base_url(request)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           f"<url><loc>{base}/</loc></url><url><loc>{base}/privacy</loc></url></urlset>")
    return Response(xml, media_type="application/xml")


@mcp.custom_route("/status", methods=["GET"])
async def status(_: Request) -> JSONResponse:
    """Checks that the live FPL data can be reached from this server."""
    try:
        m = await load_model()
        return JSONResponse({"fpl_ok": True, "next_gameweek": m.next_gw, "players": len(m.players),
                             "calls_since_start": _stats["calls"], "unique_teams": len(_stats["teams"])})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"fpl_ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=502)


@mcp.custom_route("/privacy", methods=["GET"])
async def privacy(_: Request) -> HTMLResponse:
    return HTMLResponse(
        "<h1>Privacy policy</h1>"
        "<p>This app only uses the FPL team ID you provide to read public data from the Fantasy Premier "
        "League website. We do not ask for passwords, we do not store your conversations, and we do not "
        "sell or share any data. Requests are cached in memory for up to 10 minutes and then discarded. We count tool usage anonymously (team IDs are replaced by an irreversible short hash) to improve the app.</p>")


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


app = mcp.streamable_http_app()

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
except Exception:  # pragma: no cover
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
