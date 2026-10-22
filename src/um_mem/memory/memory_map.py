"""Render a self-contained mind map of one memory workspace.

The page is a tidy left-to-right tree: the workspace root sits on the left,
blocks branch out to the right, and records hang off their block as readable
cards. Every branch gets its own vertical band and a faint group capsule, so
nothing overlaps and the hierarchy reads at a glance. Semantic similarity is
drawn as faint dashed arcs between records. The whole scene is one canvas
with no external assets and no build step, so it opens offline.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from um_mem.api import MemoryClient

#: Records are linked when their embeddings are at least this close. The
#: threshold keeps the graph readable instead of turning it into a hairball.
SEMANTIC_LINK_FLOOR = 0.80

#: How many strongest semantic links a single record may draw. Without the cap
#: a small, homogeneous store links every record to every other one.
SEMANTIC_LINKS_PER_RECORD = 2

_STYLE = """:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel-2: #f0f2f8;
  --line: #e2e6f0;
  --text: #1e2433;
  --muted: #6f7789;
  --accent: #5b5bd6;
  --accent-soft: #c9cbf7;
  --good: #15803d;
  --warn: #b45309;
  --bad: #b91c1c;
  --shadow: 0 1px 2px rgba(20, 30, 60, .06), 0 8px 22px rgba(20, 30, 60, .06);
}
body[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0f1118;
  --panel: #161a23;
  --panel-2: #1e2330;
  --line: #2a3040;
  --text: #e8eaf0;
  --muted: #8b93a7;
  --accent: #a882ff;
  --accent-soft: #4b3f7a;
  --good: #3fb950;
  --warn: #d29922;
  --bad: #f85149;
  --shadow: 0 1px 2px rgba(0, 0, 0, .45), 0 10px 26px rgba(0, 0, 0, .35);
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.55 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  overflow: hidden;
  display: flex; flex-direction: column;
}
button, input, select { font: inherit; color: inherit; }
header {
  flex: 0 0 auto; display: flex; align-items: center; gap: 10px 16px; flex-wrap: wrap;
  min-height: 58px; padding: 9px 16px; border-bottom: 1px solid var(--line);
  background: var(--panel); position: relative; z-index: 3;
}
.brand { display: flex; flex-direction: column; min-width: 0; }
header h1 { margin: 0; font-size: 15px; font-weight: 650; letter-spacing: .1px; white-space: nowrap; }
header .sub { color: var(--muted); font-size: 11px; max-width: 320px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.tools { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.tools input[type="search"] {
  width: 220px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 9px; padding: 6px 11px; outline: none; transition: border-color .15s, box-shadow .15s;
}
.tools input[type="search"]:focus {
  border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
}
.tools select, .tools button, .chip-toggle {
  background: var(--panel); border: 1px solid var(--line); border-radius: 9px;
  padding: 6px 10px; cursor: pointer; transition: border-color .15s, background .15s;
}
.tools button:hover, .chip-toggle:hover, .tools select:hover { border-color: var(--accent); }
.chip-toggle { display: inline-flex; align-items: center; gap: 6px; user-select: none; color: var(--muted); }
.chip-toggle input { accent-color: var(--accent); margin: 0; }
.stats { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
.stat {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 4px 9px; font-size: 11px; color: var(--muted); white-space: nowrap;
}
.stat b { color: var(--text); font-size: 12.5px; font-weight: 650; margin-right: 4px; }
main { flex: 1 1 auto; min-height: 0; display: flex; }
#stage { position: relative; flex: 1; min-width: 0; overflow: hidden; }
#graph { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
#graph.dragging { cursor: grabbing; }
#graph.pointing { cursor: pointer; }
.hint {
  position: absolute; left: 50%; bottom: 12px; transform: translateX(-50%);
  color: var(--muted); font-size: 11px; pointer-events: none; white-space: nowrap;
  background: color-mix(in srgb, var(--panel) 82%, transparent);
  border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px;
}
#tooltip {
  position: absolute; z-index: 5; max-width: 320px; padding: 8px 11px;
  border-radius: 10px; background: var(--panel); border: 1px solid var(--line);
  box-shadow: var(--shadow); pointer-events: none;
  opacity: 0; transform: translateY(4px); transition: opacity .12s, transform .12s;
}
#tooltip.on { opacity: 1; transform: translateY(0); }
#tooltip .t { font-weight: 600; font-size: 12.5px; margin-bottom: 2px; }
#tooltip .m { color: var(--muted); font-size: 11px; }
#tooltip .d { color: var(--text); font-size: 11px; line-height: 1.45; margin-top: 5px; opacity: .86; }
.matches {
  position: absolute; top: 12px; left: 14px; padding: 4px 10px; border-radius: 999px;
  background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow);
  color: var(--accent); font-size: 11.5px; opacity: 0; transition: opacity .16s;
  pointer-events: none;
}
.matches.on { opacity: 1; }
aside {
  width: 380px; flex: 0 0 380px; border-left: 1px solid var(--line);
  background: var(--panel); overflow-y: auto; padding: 18px 18px 48px;
  scrollbar-width: thin;
}
aside .crumbs { color: var(--muted); font-size: 11px; margin-bottom: 6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
aside h2 { margin: 0 0 8px; font-size: 17px; line-height: 1.35; word-break: break-word; letter-spacing: -.1px; }
aside h3 { margin: 20px 0 8px; font-size: 10.5px; letter-spacing: .8px;
  text-transform: uppercase; color: var(--muted); }
aside p { margin: 0; }
.kind {
  display: inline-block; font-size: 10px; padding: 2px 8px; border-radius: 999px;
  background: color-mix(in srgb, var(--c, #6b7280) 14%, transparent);
  color: var(--c, #6b7280); font-weight: 700; letter-spacing: .3px; margin-bottom: 9px;
}
.chips { display: flex; flex-wrap: wrap; gap: 5px; margin: 0 0 12px; }
.pill {
  font-size: 10.5px; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--muted); background: var(--panel-2);
}
.pill.fresh { color: var(--good); border-color: color-mix(in srgb, var(--good) 30%, transparent); }
.pill.aging { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 30%, transparent); }
.pill.stale { color: var(--bad); border-color: color-mix(in srgb, var(--bad) 30%, transparent); }
.text { color: color-mix(in srgb, var(--text) 88%, var(--muted)); white-space: pre-wrap; overflow-wrap: anywhere; font-size: 13px; }
.steps { margin: 8px 0 0; padding-left: 20px; color: color-mix(in srgb, var(--text) 88%, var(--muted)); font-size: 13px; }
.steps li { margin-bottom: 6px; overflow-wrap: anywhere; }
.note { margin-top: 12px; padding: 9px 11px; border-radius: 9px; background: var(--panel-2);
  border: 1px solid var(--line); color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.note b { color: var(--text); }
.action {
  margin-top: 14px; display: inline-flex; align-items: center; gap: 6px;
  background: var(--panel); border: 1px solid var(--line); color: var(--text);
  border-radius: 9px; padding: 6px 11px; cursor: pointer; font-size: 12px;
}
.action:hover { border-color: var(--accent); color: var(--accent); }
.reclist, .blocklist, .rel { display: flex; flex-direction: column; gap: 1px; }
.reclist a, .rel a, .blocklist a {
  display: block; padding: 7px 9px; border-radius: 8px; color: var(--text);
  text-decoration: none; cursor: pointer; font-size: 12.5px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  transition: background .12s;
}
.reclist a:hover, .rel a:hover, .blocklist a:hover { background: var(--panel-2); }
.reclist a i, .blocklist a i {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 8px; vertical-align: 1px;
}
.rel .score { float: right; color: var(--accent); font-size: 11px; }
.empty { color: var(--muted); font-size: 12.5px; }
.legend-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin-top: 4px; }
.legend-grid span { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }
.legend-grid i { width: 8px; height: 8px; border-radius: 50%; }
@media (max-width: 980px) {
  aside { width: 320px; flex-basis: 320px; }
  .stats { display: none; }
  .hint { display: none; }
}
@media (max-width: 760px) {
  aside { display: none; }
  .tools input[type="search"] { width: 160px; }
}
"""

_SCRIPT = r"""(function () {
  "use strict";

  const RAW = JSON.parse(document.getElementById("map-data").textContent);
  const canvas = document.getElementById("graph");
  const ctx = canvas.getContext("2d", { alpha: false });
  const stage = document.getElementById("stage");
  const tooltip = document.getElementById("tooltip");
  const detail = document.getElementById("detail");
  const search = document.getElementById("q");
  const kindSelect = document.getElementById("kind");
  const linksToggle = document.getElementById("links");
  const themeButton = document.getElementById("theme");
  const matchesEl = document.getElementById("matches");

  const THEMES = {
    light: {
      bg1: "#ffffff", bg2: "#eef1f8", grid: "rgba(70, 90, 130, .10)",
      tree: "#ccd3e2", treeHi: "#7c6cf0",
      chip: "#ffffff", chipAlt: "#f4f6fc", border: "#e2e6f0",
      text: "#242938", muted: "#7c8496",
      accent: "#5b5bd6", accentSoft: "#b9baf5",
      semantic: "#c3b7f2", shadow: "rgba(28, 38, 70, .13)", dim: 0.4,
      group: "rgba(91, 91, 214, .045)"
    },
    dark: {
      bg1: "#1a1f2e", bg2: "#0d0f16", grid: "rgba(140, 155, 190, .10)",
      tree: "#363d4e", treeHi: "#a882ff",
      chip: "#1c202c", chipAlt: "#232838", border: "#343b4d",
      text: "#e8eaf0", muted: "#8b93a7",
      accent: "#a882ff", accentSoft: "#6f5ac2",
      semantic: "#6b5cc4", shadow: "rgba(0, 0, 0, .5)", dim: 0.36,
      group: "rgba(168, 130, 255, .055)"
    }
  };
  const KIND_COLORS = {
    light: {
      project_fact: "#4f7cf7", procedure: "#8b5cf6", observation: "#0ea5b7",
      preference: "#d97706", decision: "#16a34a", directive: "#dc2626"
    },
    dark: {
      project_fact: "#6ea8fe", procedure: "#c792ea", observation: "#56d4dd",
      preference: "#f0b96b", decision: "#7ee787", directive: "#ff9b9b"
    }
  };
  const FALLBACK = { light: "#64748b", dark: "#93a1b8" };

  const GROUP_GAP = 14;
  const COLUMN_GAP = 70;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const ease = (rate) => Math.min(1, rate);

  const state = {
    nodes: [], links: [], byId: new Map(),
    view: { x: 0, y: 0, k: 1 },
    target: { x: 0, y: 0, k: 1 },
    W: 1, H: 1, dpr: 1,
    t: 0, reveal: 0, last: performance.now(),
    hover: null, selected: null,
    filter: "", query: "", showMeta: false, showLinks: true,
    theme: "light",
    collapsed: new Set(),
    drag: null, pan: null,
    pointer: { x: 0, y: 0, down: false, moved: false },
    userView: false, dirty: true, matches: 0, pulseAt: -10
  };

  function theme() { return THEMES[state.theme]; }
  function kindColor(kind) {
    return (KIND_COLORS[state.theme] || {})[kind] || FALLBACK[state.theme];
  }
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"]/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
    }[ch]));
  }
  function truncate(text, limit) {
    const value = String(text || "");
    return value.length > limit ? value.slice(0, limit - 1) + "\u2026" : value;
  }

  function fitText(text, font, maxWidth) {
    ctx.font = font;
    if (ctx.measureText(text).width <= maxWidth) return text;
    let value = text;
    while (value.length > 4 && ctx.measureText(value + "\u2026").width > maxWidth) {
      value = value.slice(0, -1);
    }
    return value + "\u2026";
  }

  function hasChildren(node) {
    return state.nodes.some((other) => other.parent === node.id);
  }

  function sizeNode(node) {
    if (node.type === "block") {
      const font = "600 13px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      const small = "10.5px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      const isRoot = !node.parent;
      const maxText = isRoot ? 180 : 230;
      node.text = fitText(node.label || "(unnamed)", font, maxText);
      ctx.font = font;
      const textW = ctx.measureText(node.text).width;
      if (isRoot) {
        node.subtext = "";
        node.chipW = Math.max(132, textW + 44);
        node.chipH = 42;
      } else {
        const tag = String(node.node_kind || "").trim();
        let detail = String(node.synopsis || node.description || "").replace(/\s+/g, " ").trim();
        const sentence = detail.match(/^.+?[.!?](\s|$)/);
        if (sentence) detail = sentence[0].trim();
        const bits = [];
        if (tag) bits.push(tag.toUpperCase());
        if (node.index_only) bits.push("INDEX ONLY");
        if (detail) bits.push(detail);
        node.subtext = bits.length ? fitText(bits.join(" \u00b7 "), small, 200) : "";
        ctx.font = small;
        const subW = node.subtext ? ctx.measureText(node.subtext).width : 0;
        const reserved = 30 + (node.records > 0 ? 42 : 0) + (hasChildren(node) ? 18 : 0);
        node.chipW = Math.max(120, textW + reserved, subW + 30);
        node.chipH = node.subtext ? 46 : 34;
      }
    } else {
      const font = "12.5px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      node.text = fitText(node.label || "(untitled)", font, 300);
      ctx.font = font;
      node.chipW = Math.max(96, ctx.measureText(node.text).width + 40);
      node.chipH = 28;
    }
  }

  function computeVisibility() {
    const byId = state.byId;
    for (const node of state.nodes) {
      const metaHidden = Boolean(node.meta) && !state.showMeta;
      const kindHidden = node.type === "record" && Boolean(state.filter) && node.kind !== state.filter;
      node.baseHidden = metaHidden || kindHidden;
    }
    const collapsed = state.collapsed;
    const hiddenByAncestor = (node) => {
      let parent = byId.get(node.parent);
      while (parent) {
        if (parent.baseHidden || collapsed.has(parent.id)) return true;
        parent = byId.get(parent.parent);
      }
      return false;
    };
    for (const node of state.nodes) {
      node.hidden = node.baseHidden || hiddenByAncestor(node);
    }
    for (const link of state.links) {
      link.hidden = link.a.hidden || link.b.hidden;
    }
    if (state.hover && state.hover.hidden) state.hover = null;
    if (state.selected && state.selected.hidden) state.selected = null;
  }

  function relayout() {
    computeVisibility();
    const kids = new Map();
    for (const node of state.nodes) {
      if (node.hidden) continue;
      const key = node.parent && state.byId.has(node.parent) ? node.parent : "";
      if (!kids.has(key)) kids.set(key, []);
      kids.get(key).push(node);
    }
    for (const list of kids.values()) {
      list.sort((left, right) => {
        if (left.type !== right.type) return left.type === "block" ? -1 : 1;
        return left.label.localeCompare(right.label, undefined, { numeric: true });
      });
    }
    const roots = kids.get("") || [];
    const depthOf = new Map();
    const markDepth = (node, depth) => {
      depthOf.set(node.id, depth);
      for (const child of kids.get(node.id) || []) markDepth(child, depth + 1);
    };
    for (const root of roots) markDepth(root, 0);

    const heights = new Map();
    const measure = (node) => {
      const children = kids.get(node.id) || [];
      const own = node.type === "block" ? Math.max(52, node.chipH + 10) : 40;
      if (!children.length) {
        heights.set(node.id, own);
        return own;
      }
      const total = children.reduce((sum, child) => sum + measure(child), 0) +
        GROUP_GAP * (children.length - 1);
      const height = Math.max(own, total);
      heights.set(node.id, height);
      return height;
    };
    roots.forEach(measure);

    const place = (node, top) => {
      const children = kids.get(node.id) || [];
      const height = heights.get(node.id) || 40;
      node._top = top;
      node._bottom = top + height;
      if (!children.length) {
        node.ty = top + height / 2;
        return;
      }
      let cursor = top;
      for (const child of children) {
        place(child, cursor);
        cursor += heights.get(child.id) + GROUP_GAP;
      }
      const first = children[0];
      const last = children[children.length - 1];
      node.ty = (first.ty + last.ty) / 2;
    };
    let cursor = 0;
    for (const root of roots) {
      place(root, cursor);
      cursor += heights.get(root.id) + GROUP_GAP * 2;
    }
    const shift = (cursor - GROUP_GAP * 2) / 2;

    const visible = state.nodes.filter((node) => !node.hidden);
    for (const node of visible) {
      node.depth = depthOf.get(node.id) || 0;
      node.ty -= shift;
    }
    const half = new Map();
    for (const node of visible) {
      const depth = node.depth;
      half.set(depth, Math.max(half.get(depth) || 0, node.chipW / 2));
    }
    const columns = new Map();
    const depths = [...half.keys()].sort((left, right) => left - right);
    let previous = 0;
    for (const depth of depths) {
      if (depth === 0) {
        columns.set(0, 0);
        continue;
      }
      const x = (columns.get(previous) || 0) + (half.get(previous) || 0) +
        (half.get(depth) || 0) + COLUMN_GAP;
      columns.set(depth, x);
      previous = depth;
    }
    for (const node of visible) {
      node.tx = (columns.get(node.depth) || 0) + (node.dx || 0);
      node.ty += node.dy || 0;
    }
    state.dirty = true;
  }

  function bounds(list) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const node of list) {
      if (node.hidden) continue;
      const x = node.tx == null ? node.x : node.tx;
      const y = node.ty == null ? node.y : node.ty;
      minX = Math.min(minX, x - node.chipW / 2);
      maxX = Math.max(maxX, x + node.chipW / 2);
      minY = Math.min(minY, y - node.chipH / 2);
      maxY = Math.max(maxY, y + node.chipH / 2);
    }
    if (!isFinite(minX)) return null;
    return { minX, maxX, minY, maxY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2,
      w: Math.max(80, maxX - minX), h: Math.max(60, maxY - minY) };
  }

  function subtree(node) {
    const found = [];
    const walk = (current) => {
      found.push(current);
      for (const child of state.nodes) {
        if (child.parent === current.id && !child.hidden) walk(child);
      }
    };
    walk(node);
    return found;
  }

  function branchSet(node) {
    const set = new Set();
    let current = node;
    while (current) {
      set.add(current.id);
      current = state.byId.get(current.parent) || null;
    }
    for (const item of subtree(node)) set.add(item.id);
    if (node.type === "record") {
      for (const link of state.links) {
        if (link.kind !== "semantic") continue;
        if (link.a.id === node.id) set.add(link.b.id);
        if (link.b.id === node.id) set.add(link.a.id);
      }
    }
    return set;
  }

  function fit(subset, padding) {
    const list = (subset || state.nodes).filter((node) => !node.hidden);
    const box = bounds(list);
    if (!box) return;
    const margin = padding == null ? 96 : padding;
    const k = clamp(
      Math.min((state.W - margin * 2) / box.w, (state.H - margin * 2) / box.h),
      0.3, 1.35
    );
    state.target.k = k;
    state.target.x = box.cx;
    state.target.y = box.cy;
    if (!subset) state.userView = false;
  }

  function layoutKey() {
    return "um-mem-map-layout:" + (RAW.workspace_id || "default");
  }

  function loadLayout() {
    try {
      const raw = JSON.parse(localStorage.getItem(layoutKey()) || "{}");
      for (const node of state.nodes) {
        const offset = raw[node.id];
        if (Array.isArray(offset) && offset.length === 2) {
          node.dx = Number(offset[0]) || 0;
          node.dy = Number(offset[1]) || 0;
        }
      }
    } catch (error) { /* file:// or stale data */ }
  }

  function saveLayout() {
    try {
      const raw = {};
      for (const node of state.nodes) {
        if (node.dx || node.dy) raw[node.id] = [node.dx, node.dy];
      }
      if (Object.keys(raw).length) localStorage.setItem(layoutKey(), JSON.stringify(raw));
      else localStorage.removeItem(layoutKey());
    } catch (error) { /* file:// */ }
  }

  function resetLayout() {
    for (const node of state.nodes) {
      node.dx = 0;
      node.dy = 0;
    }
    saveLayout();
    relayout();
    fit();
    state.dirty = true;
  }

  function focusNode(node) {
    if (node.type === "block") {
      const box = bounds(subtree(node));
      if (box) {
        const margin = 110;
        const k = clamp(
          Math.min((state.W - margin * 2) / box.w, (state.H - margin * 2) / box.h),
          0.45, 1.45
        );
        state.target.k = k;
        state.target.x = box.cx;
        state.target.y = box.cy;
      }
    } else {
      state.target.k = clamp(Math.max(state.target.k, 1.1), 0.3, 2.4);
      state.target.x = node.tx == null ? node.x : node.tx;
      state.target.y = node.ty == null ? node.y : node.ty;
    }
    state.userView = true;
  }

  function screenToWorld(sx, sy) {
    return {
      x: state.view.x + (sx - state.W / 2) / state.view.k,
      y: state.view.y + (sy - state.H / 2) / state.view.k
    };
  }

  function zoomAt(sx, sy, factor) {
    const before = screenToWorld(sx, sy);
    const k = clamp(state.target.k * factor, 0.28, 2.6);
    state.target.k = k;
    state.target.x = before.x - (sx - state.W / 2) / k;
    state.target.y = before.y - (sy - state.H / 2) / k;
    state.userView = true;
  }

  function pick(sx, sy) {
    const point = screenToWorld(sx, sy);
    for (let index = state.nodes.length - 1; index >= 0; index--) {
      const node = state.nodes[index];
      if (node.hidden || node.appear < 0.45) continue;
      if (Math.abs(point.x - node.x) <= node.chipW / 2 + 3 &&
          Math.abs(point.y - node.y) <= node.chipH / 2 + 3) {
        return node;
      }
    }
    return null;
  }

  function toggleCollapse(node, force) {
    if (!hasChildren(node)) return;
    if (state.collapsed.has(node.id) && force !== true) {
      state.collapsed.delete(node.id);
    } else {
      state.collapsed.add(node.id);
    }
    relayout();
    state.dirty = true;
    if (state.selected === node) renderDetail();
  }
  function focusSet() {
    const anchor = state.hover || state.selected;
    return anchor ? branchSet(anchor) : null;
  }

  function matchesQuery(node) {
    return !state.query || (node.hay || "").includes(state.query);
  }

  function roundRect(x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function treeLink(a, b) {
    const startX = a.x + a.chipW / 2;
    const endX = b.x - b.chipW / 2;
    const midX = (startX + endX) / 2;
    ctx.beginPath();
    ctx.moveTo(startX, a.y);
    ctx.bezierCurveTo(midX, a.y, midX, b.y, endX, b.y);
  }

  function drawGroups(alpha) {
    const th = theme();
    for (const node of state.nodes) {
      if (node.hidden || node.depth !== 1 || node.meta || !hasChildren(node)) continue;
      const children = state.nodes.filter((item) => item.parent === node.id && !item.hidden);
      if (children.length < 2) continue;
      let top = Infinity;
      let bottom = -Infinity;
      let left = Infinity;
      let right = -Infinity;
      const walk = (item) => {
        top = Math.min(top, item.y - item.chipH / 2 - 9);
        bottom = Math.max(bottom, item.y + item.chipH / 2 + 9);
        left = Math.min(left, item.x - item.chipW / 2 - 9);
        right = Math.max(right, item.x + item.chipW / 2 + 9);
        for (const child of state.nodes) {
          if (child.parent === item.id && !child.hidden) walk(child);
        }
      };
      for (const child of children) walk(child);
      roundRect(left, top, right - left, bottom - top, 18);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = th.group;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function similarArc(a, b) {
    const ax = a.x + a.chipW / 2 + 5;
    const bx = b.x + b.chipW / 2 + 5;
    const bow = 16 + Math.min(44, Math.abs(b.y - a.y) * 0.2);
    ctx.beginPath();
    ctx.moveTo(ax, a.y);
    ctx.bezierCurveTo(ax + bow, a.y, bx + bow, b.y, bx, b.y);
  }

  function render() {
    const th = theme();
    const k = state.view.k;
    const W = state.W, H = state.H;
    ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
    const sky = ctx.createRadialGradient(W * 0.5, H * 0.08, 30, W * 0.5, H * 0.6, Math.max(W, H) * 0.95);
    sky.addColorStop(0, th.bg1);
    sky.addColorStop(1, th.bg2);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);

    const spacing = 36 * k;
    if (spacing > 14) {
      const startX = ((W / 2 - state.view.x * k) % spacing + spacing) % spacing;
      const startY = ((H / 2 - state.view.y * k) % spacing + spacing) % spacing;
      ctx.fillStyle = th.grid;
      for (let x = startX; x < W; x += spacing) {
        for (let y = startY; y < H; y += spacing) {
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }

    ctx.save();
    ctx.translate(W / 2, H / 2);
    ctx.scale(k, k);
    ctx.translate(-state.view.x, -state.view.y);

    const focus = focusSet();
    const alphaOf = (node) => {
      let value = !focus || focus.has(node.id) ? 1 : th.dim;
      if (state.query && node.type === "record" && !matchesQuery(node)) value *= 0.16;
      return value * (node.appear == null ? 1 : node.appear);
    };
    const line = 1 / k;

    drawGroups(Math.min(1, state.t * 2) * 0.9);

    if (state.showLinks) {
      ctx.strokeStyle = th.semantic;
      ctx.fillStyle = th.semantic;
      ctx.lineWidth = 1.1 * line;
      ctx.setLineDash([4 * line, 5 * line]);
      for (const link of state.links) {
        if (link.kind !== "semantic" || link.draw === false || link.hidden) continue;
        ctx.globalAlpha = 0.6 * Math.min(alphaOf(link.a), alphaOf(link.b));
        similarArc(link.a, link.b);
        ctx.stroke();
        ctx.setLineDash([]);
        for (const node of [link.a, link.b]) {
          ctx.beginPath();
          ctx.arc(node.x + node.chipW / 2 + 5, node.y, 2.2 * line, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.setLineDash([4 * line, 5 * line]);
      }
      ctx.setLineDash([]);
    }

    for (const link of state.links) {
      if (link.kind === "semantic" || link.hidden) continue;
      const both = !focus || (focus.has(link.a.id) && focus.has(link.b.id));
      const one = focus && (focus.has(link.a.id) || focus.has(link.b.id));
      ctx.globalAlpha = Math.min(alphaOf(link.a), alphaOf(link.b)) * (one ? 1 : 0.35);
      if (link.kind === "tree") {
        ctx.strokeStyle = both && focus ? th.treeHi : th.tree;
        ctx.lineWidth = (both && focus ? 2 : 1.5) * line;
      } else {
        ctx.strokeStyle = both && focus ? th.treeHi : th.tree;
        ctx.lineWidth = (both && focus ? 1.6 : 1.1) * line;
      }
      treeLink(link.a, link.b);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const ordered = state.nodes
      .filter((node) => node.appear == null || node.appear > 0.01)
      .sort((left, right) => (left.depth || 0) - (right.depth || 0));
    for (const node of ordered) drawChip(node, alphaOf(node));

    if (state.selected) {
      drawPulse(state.selected);
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function drawPulse(node) {
    if (node.hidden) return;
    const th = theme();
    const age = state.t - state.pulseAt;
    const pulse = Math.max(0, 1 - age / 1.8);
    if (pulse <= 0.01) return;
    const grow = (1 - pulse) * 10;
    ctx.globalAlpha = 0.5 * pulse;
    ctx.strokeStyle = th.accent;
    ctx.lineWidth = 1.6;
    roundRect(
      node.x - node.chipW / 2 - 5 - grow,
      node.y - node.chipH / 2 - 5 - grow,
      node.chipW + 10 + grow * 2,
      node.chipH + 10 + grow * 2,
      14
    );
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawChip(node, alpha) {
    if (alpha < 0.02) return;
    const th = theme();
    const selected = state.selected === node;
    const hovered = state.hover === node;
    const isRoot = node.type === "block" && !node.parent;
    const w = node.chipW, h = node.chipH;
    const x = node.x - w / 2, y = node.y - h / 2;
    const radius = node.type === "block" ? 10 : 9;
    const hover = node.hoverAmt || 0;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(node.x, node.y);
    ctx.scale(1 + hover * 0.02, 1 + hover * 0.02);
    ctx.translate(-node.x, -node.y);

    if (state.query && node.type === "record" && matchesQuery(node)) {
      ctx.globalAlpha = alpha * 0.5;
      ctx.strokeStyle = th.accent;
      ctx.lineWidth = 3;
      roundRect(x - 4, y - 4, w + 8, h + 8, radius + 4);
      ctx.stroke();
      ctx.globalAlpha = alpha;
    }

    ctx.shadowColor = th.shadow;
    ctx.shadowBlur = selected || hovered ? 20 : 10;
    ctx.shadowOffsetY = selected || hovered ? 6 : 2;
    roundRect(x, y, w, h, radius);
    if (isRoot) {
      const fill = ctx.createLinearGradient(x, y, x + w, y + h);
      fill.addColorStop(0, "#5f5fe8");
      fill.addColorStop(1, "#3f3fb8");
      ctx.fillStyle = fill;
    } else {
      ctx.fillStyle = node.type === "block" ? th.chipAlt : th.chip;
    }
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    ctx.lineWidth = selected ? 1.8 : 1;
    ctx.strokeStyle = selected ? th.accent : hovered ? th.accentSoft : th.border;
    ctx.stroke();

    ctx.textBaseline = "middle";
    if (isRoot) {
      ctx.font = "600 14px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(node.text, node.x, node.y + 0.5);
      ctx.restore();
      return;
    }

    if (node.type === "block") {
      const titleY = node.y + (node.subtext ? -7 : 0.5);
      ctx.beginPath();
      ctx.arc(x + 16, titleY, 4, 0, Math.PI * 2);
      ctx.fillStyle = th.accent;
      ctx.fill();
      ctx.textAlign = "left";
      ctx.font = "600 13px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = th.text;
      ctx.fillText(node.text, x + 28, titleY);
      if (node.records > 0) {
        ctx.font = "11px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
        ctx.fillStyle = th.muted;
        ctx.textAlign = "right";
        ctx.fillText(node.records + " rec", x + w - (hasChildren(node) ? 28 : 12), titleY);
      }
      if (hasChildren(node)) {
        const cx = x + w - 14;
        ctx.strokeStyle = th.muted;
        ctx.lineWidth = 1.5;
        ctx.lineCap = "round";
        ctx.beginPath();
        if (state.collapsed.has(node.id)) {
          ctx.moveTo(cx - 2.5, titleY - 4.5);
          ctx.lineTo(cx + 2, titleY);
          ctx.lineTo(cx - 2.5, titleY + 4.5);
        } else {
          ctx.moveTo(cx - 4.5, titleY - 2);
          ctx.lineTo(cx, titleY + 2.5);
          ctx.lineTo(cx + 4.5, titleY - 2);
        }
        ctx.stroke();
        ctx.lineCap = "butt";
      }
      if (node.subtext) {
        ctx.textAlign = "left";
        ctx.font = "10.5px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
        ctx.fillStyle = th.muted;
        ctx.fillText(node.subtext, x + 28, node.y + 9);
      }
    } else {
      ctx.beginPath();
      ctx.arc(x + 15, node.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = kindColor(node.kind);
      ctx.fill();
      ctx.textAlign = "left";
      ctx.font = "12.5px ui-sans-serif, system-ui, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = th.text;
      ctx.fillText(node.text, x + 25, node.y + 0.5);
    }
    ctx.restore();
  }
  function renderDetail() {
    const node = state.selected;
    if (!node) {
      const blocks = state.nodes.filter((item) => item.type === "block" && item.parent && !item.hidden && !item.meta);
      const list = blocks.map((block) =>
        '<a data-go="' + esc(block.id) + '"><i style="background:' +
        (block.index_only ? "#8b8bd9" : "#7c6cf0") + '"></i>' +
        esc(block.label) + " (" + block.records + ")</a>").join("");
      const kinds = [...new Set(state.nodes.filter((item) => item.type === "record").map((item) => item.kind))]
        .filter(Boolean).sort();
      detail.innerHTML =
        "<h2>Memory graph</h2>" +
        '<p class="empty">The root sits on the left, branches grow to the right, and records hang off each branch. Click a card to read it, drag a card to move it (its records follow), click the chevron to fold a branch, scroll to zoom.</p>' +
        '<button class="action" data-reset="1">Reset card layout</button>' +
        "<h3>Branches</h3>" +
        '<div class="blocklist">' + (list || '<span class="empty">No blocks yet.</span>') + "</div>" +
        "<h3>Record kinds</h3>" +
        '<div class="legend-grid">' + kinds.map((kind) =>
          '<span><i style="background:' + kindColor(kind) + '"></i>' + esc(kind) + "</span>").join("") +
        '<span><i style="background:transparent;border:1px dashed ' + theme().accent + '"></i>semantic link</span>' +
        "</div>";
      return;
    }
    if (node.type === "block") {
      const children = state.nodes.filter((item) => item.type === "record" && item.parent === node.id);
      const counts = {};
      for (const child of children) {
        const key = child.kind || "untyped";
        counts[key] = (counts[key] || 0) + 1;
      }
      const collapsed = state.collapsed.has(node.id);
      detail.innerHTML =
        '<div class="crumbs">' + esc((node.path_names || []).join(" / ")) + "</div>" +
        "<h2>" + esc(node.label) + "</h2>" +
        '<div class="chips">' +
        '<span class="pill">' + children.length + " records</span>" +
        (node.index_only ? '<span class="pill">index only</span>' : '<span class="pill fresh">writable</span>') +
        (node.meta ? '<span class="pill">meta</span>' : "") +
        Object.entries(counts).map(([kind, count]) =>
          '<span class="pill" style="color:' + kindColor(kind) + '">' + esc(kind) + " " + count + "</span>").join("") +
        "</div>" +
        (node.description ? '<p class="text">' + esc(node.description) + "</p>" : "") +
        "<h3>Records</h3>" +
        '<div class="reclist">' + (children.length
          ? children.map((child) =>
              '<a data-go="' + esc(child.id) + '"><i style="background:' + kindColor(child.kind) +
              '"></i>' + esc(child.label) + "</a>").join("")
          : '<span class="empty">This block has no records.</span>') + "</div>" +
        (node.parent && hasChildren(node)
          ? '<button class="action" data-collapse="' + esc(node.id) + '">' +
            (collapsed ? "Expand records" : "Collapse records") + "</button>"
          : "");
      return;
    }
    const related = [];
    for (const link of state.links) {
      if (link.kind !== "semantic") continue;
      if (link.a.id === node.id) related.push({ node: link.b, weight: link.weight || 0 });
      else if (link.b.id === node.id) related.push({ node: link.a, weight: link.weight || 0 });
    }
    related.sort((left, right) => right.weight - left.weight);
    detail.innerHTML =
      '<span class="kind" style="--c:' + kindColor(node.kind) + '">' + esc(node.kind || "record") + "</span>" +
      "<h2>" + esc(node.label) + "</h2>" +
      '<div class="chips">' +
      (node.age ? '<span class="pill">' + esc(node.age) + "</span>" : "") +
      (node.status ? '<span class="pill">' + esc(node.status) + "</span>" : "") +
      (node.freshness ? '<span class="pill ' + esc(node.freshness) + '">' + esc(node.freshness) + "</span>" : "") +
      (node.tags || []).map((tag) => '<span class="pill">#' + esc(tag) + "</span>").join("") +
      "</div>" +
      '<p class="text">' + esc(node.text || "") + "</p>" +
      (node.steps && node.steps.length
        ? "<h3>Steps</h3><ol class=\"steps\">" +
          node.steps.map((step) => "<li>" + esc(step) + "</li>").join("") + "</ol>"
        : "") +
      (node.verify ? '<div class="note"><b>Verify:</b> ' + esc(node.verify) + "</div>" : "") +
      (node.parent && state.byId.get(node.parent)
        ? '<h3>Block</h3><div class="blocklist"><a data-go="' + esc(node.parent) + '"><i style="background:' +
          theme().accent + '"></i>' + esc(state.byId.get(node.parent).label) + "</a></div>"
        : "") +
      (related.length
        ? "<h3>Related</h3><div class=\"rel\">" + related.map((item) =>
            '<a data-go="' + esc(item.node.id) + '"><span class="score">' +
            Math.round(item.weight * 100) + "%</span>" + esc(item.node.label) + "</a>").join("") + "</div>"
        : "");
  }

  function select(node) {
    state.selected = node;
    state.pulseAt = state.t;
    if (node && node.type === "block") focusNode(node);
    renderDetail();
    state.dirty = true;
  }

  function showTooltip(node, sx, sy) {
    if (!node) {
      tooltip.classList.remove("on");
      return;
    }
    const bits = [];
    let body = "";
    let extra = "";
    if (node.type === "block") {
      if (node.node_kind) bits.push(node.node_kind);
      bits.push((node.records || 0) + " records");
      if (node.index_only) bits.push("index only");
      if (node.meta) bits.push("meta");
      body = String(node.synopsis || node.description || "").replace(/\s+/g, " ").trim();
      const children = state.nodes.filter((item) => item.type === "record" && item.parent === node.id);
      if (children.length) {
        extra = children.slice(0, 3).map((item) => truncate(item.label, 34)).join(" \u00b7 ") +
          (children.length > 3 ? " \u2026" : "");
      }
    } else {
      bits.push(node.kind || "record");
      if (node.age) bits.push(node.age);
      if (node.freshness) bits.push(node.freshness);
      body = String(node.text || "").replace(/\s+/g, " ").trim();
    }
    const path = (node.path_names || []).slice(0, -1).join(" / ");
    tooltip.innerHTML = '<div class="t">' + esc(node.label) + "</div>" +
      '<div class="m">' + esc(bits.join(" \u00b7 ")) + "</div>" +
      (path ? '<div class="m">' + esc(path) + "</div>" : "") +
      (body ? '<div class="d">' + esc(truncate(body, 240)) + "</div>" : "") +
      (extra ? '<div class="d">' + esc(extra) + "</div>" : "");
    const width = tooltip.offsetWidth || 220;
    const height = tooltip.offsetHeight || 60;
    tooltip.style.left = Math.max(8, Math.min(state.W - width - 12, sx + 16)) + "px";
    tooltip.style.top = Math.max(8, Math.min(state.H - height - 12, sy + 16)) + "px";
    tooltip.classList.add("on");
  }

  function resize() {
    const box = stage.getBoundingClientRect();
    state.W = Math.max(1, box.width);
    state.H = Math.max(1, box.height);
    state.dpr = Math.min(2.5, window.devicePixelRatio || 1);
    canvas.width = Math.round(state.W * state.dpr);
    canvas.height = Math.round(state.H * state.dpr);
    canvas.style.width = state.W + "px";
    canvas.style.height = state.H + "px";
    if (!state.userView) fit();
    state.dirty = true;
  }

  function position(event) {
    const box = canvas.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const point = position(event);
    state.pointer = { x: point.x, y: point.y, down: true, moved: false };
    const world = screenToWorld(point.x, point.y);
    const node = pick(point.x, point.y);
    state.drag = null;
    state.pan = null;
    if (node) {
      const chevron = node.type === "block" && node.parent && hasChildren(node) &&
        world.x > node.x + node.chipW / 2 - 30;
      state.drag = {
        node: node,
        chevron: chevron,
        group: chevron ? [] : subtree(node),
        lastX: world.x,
        lastY: world.y,
        moved: false
      };
    } else {
      state.pan = { x: point.x, y: point.y, vx: state.target.x, vy: state.target.y };
    }
    try { canvas.setPointerCapture(event.pointerId); } catch (error) { /* synthetic pointer */ }
    canvas.classList.add("dragging");
    tooltip.classList.remove("on");
  });

  canvas.addEventListener("pointermove", (event) => {
    const point = position(event);
    if (state.pointer.down) {
      if (Math.hypot(point.x - state.pointer.x, point.y - state.pointer.y) > 3) {
        state.pointer.moved = true;
      }
      if (state.drag) {
        if (!state.drag.moved && state.pointer.moved) state.drag.moved = true;
        if (state.drag.moved && state.drag.group.length) {
          const world = screenToWorld(point.x, point.y);
          const dx = world.x - state.drag.lastX;
          const dy = world.y - state.drag.lastY;
          for (const item of state.drag.group) {
            item.dx = (item.dx || 0) + dx;
            item.dy = (item.dy || 0) + dy;
            item.x += dx;
            item.y += dy;
            if (item.tx != null) item.tx += dx;
            if (item.ty != null) item.ty += dy;
          }
          state.drag.lastX = world.x;
          state.drag.lastY = world.y;
          state.dirty = true;
        }
      } else if (state.pan) {
        state.target.x = state.pan.vx - (point.x - state.pan.x) / state.target.k;
        state.target.y = state.pan.vy - (point.y - state.pan.y) / state.target.k;
        state.view.x = state.target.x;
        state.view.y = state.target.y;
        state.userView = true;
      }
      return;
    }
    const node = pick(point.x, point.y);
    state.hover = node;
    canvas.classList.toggle("pointing", Boolean(node));
    showTooltip(node, point.x, point.y);
  });

  function endPointer(event) {
    const moved = state.pointer.moved;
    const drag = state.drag;
    state.pointer.down = false;
    state.pan = null;
    state.drag = null;
    canvas.classList.remove("dragging");
    if (drag && drag.moved) {
      saveLayout();
      return;
    }
    if (moved) return;
    const local = position(event);
    const world = screenToWorld(local.x, local.y);
    const node = pick(local.x, local.y);
    if (drag && drag.chevron) {
      toggleCollapse(drag.node);
      return;
    }
    if (!node) {
      select(null);
      return;
    }
    if (node.type === "block" && node.parent && hasChildren(node) &&
        world.x > node.x + node.chipW / 2 - 30) {
      toggleCollapse(node);
      return;
    }
    select(node);
  }

  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", () => {
    state.hover = null;
    tooltip.classList.remove("on");
  });

  canvas.addEventListener("dblclick", (event) => {
    const point = position(event);
    const node = pick(point.x, point.y);
    if (node) select(node);
    else resetLayout();
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const point = position(event);
    zoomAt(point.x, point.y, event.deltaY < 0 ? 1.12 : 0.89);
  }, { passive: false });

  window.addEventListener("keydown", (event) => {
    const tag = event.target && event.target.tagName;
    const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
    if (event.key === "Escape") {
      if (typing) return;
      select(null);
      return;
    }
    if (typing) return;
    if (event.key === "f" || event.key === "F") fit();
    if (event.key === "0") fit();
    if (event.key === "+" || event.key === "=") zoomAt(state.W / 2, state.H / 2, 1.15);
    if (event.key === "-") zoomAt(state.W / 2, state.H / 2, 0.87);
  });

  detail.addEventListener("click", (event) => {
    const reset = event.target.closest("[data-reset]");
    if (reset) {
      resetLayout();
      return;
    }
    const go = event.target.closest("[data-go]");
    if (go) {
      const node = state.byId.get(go.getAttribute("data-go"));
      if (node) select(node);
      return;
    }
    const fold = event.target.closest("[data-collapse]");
    if (fold) {
      const node = state.byId.get(fold.getAttribute("data-collapse"));
      if (node) toggleCollapse(node);
    }
  });

  search.addEventListener("input", () => {
    state.query = (search.value || "").trim().toLowerCase();
    if (state.query && state.selected) select(null);
    state.dirty = true;
    const hits = state.query
      ? state.nodes.filter((node) => !node.hidden && node.type === "record" && matchesQuery(node))
      : [];
    state.matches = hits.length;
    if (state.query) {
      matchesEl.textContent = hits.length + (hits.length === 1 ? " matching record" : " matching records");
      matchesEl.classList.add("on");
    } else {
      matchesEl.classList.remove("on");
    }
  });

  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !state.query) return;
    const hits = state.nodes.filter((node) => !node.hidden && node.type === "record" && matchesQuery(node));
    if (hits.length) fit(hits, 80);
  });

  kindSelect.addEventListener("change", () => {
    state.filter = kindSelect.value || "";
    relayout();
    fit();
  });

  linksToggle.addEventListener("change", () => {
    state.showLinks = linksToggle.checked;
    state.dirty = true;
  });

  themeButton.addEventListener("click", () => {
    state.theme = state.theme === "light" ? "dark" : "light";
    applyTheme();
  });

  document.getElementById("fit").addEventListener("click", () => fit());

  function applyTheme() {
    document.body.dataset.theme = state.theme;
    themeButton.textContent = state.theme === "light" ? "\u263e" : "\u2600";
    themeButton.title = state.theme === "light" ? "Dark theme" : "Light theme";
    try { localStorage.setItem("um-mem-map-theme", state.theme); } catch (error) { /* file:// */ }
    state.dirty = true;
  }

  function frame(now) {
    const delta = Math.min(64, now - state.last) || 16;
    state.last = now;
    state.t += delta / 1000;
    state.reveal = Math.min(1, state.reveal + delta / 650);

    let moving = state.reveal < 1;
    const follow = Math.min(1, delta / 140);
    for (const node of state.nodes) {
      if (node.tx != null) {
        const dx = node.tx - node.x, dy = node.ty - node.y;
        if (Math.abs(dx) > 0.2 || Math.abs(dy) > 0.2) moving = true;
        node.x += dx * follow;
        node.y += dy * follow;
      }
      const wantedAppear = node.hidden ? 0 : 1;
      const appear = node.appear == null ? 0 : node.appear;
      if (Math.abs(wantedAppear - appear) > 0.01) moving = true;
      node.appear = appear + (wantedAppear - appear) * Math.min(1, delta / 160);
      const wantedHover = node === state.hover || node === state.selected ? 1 : 0;
      const hover = node.hoverAmt == null ? 0 : node.hoverAmt;
      if (Math.abs(wantedHover - hover) > 0.01) moving = true;
      node.hoverAmt = hover + (wantedHover - hover) * Math.min(1, delta / 110);
    }

    const camera = Math.min(1, delta / 160);
    state.view.x += (state.target.x - state.view.x) * camera;
    state.view.y += (state.target.y - state.view.y) * camera;
    state.view.k += (state.target.k - state.view.k) * camera;
    const cameraMoving =
      Math.abs(state.target.x - state.view.x) > 0.05 ||
      Math.abs(state.target.y - state.view.y) > 0.05 ||
      Math.abs(state.target.k - state.view.k) > 0.0004;

    const pulsing = state.selected && state.t - state.pulseAt < 1.9;
    if (moving || cameraMoving || state.dirty || pulsing) {
      render();
      state.dirty = false;
    }
    requestAnimationFrame(frame);
  }

  function boot() {
    try {
      const saved = localStorage.getItem("um-mem-map-theme");
      if (saved === "dark" || saved === "light") state.theme = saved;
    } catch (error) { /* file:// */ }
    applyTheme();
    resize();

    const nodes = RAW.nodes.map((raw) => Object.assign({}, raw, {
      x: 0, y: 0, tx: 0, ty: 0, dx: 0, dy: 0, angle: 0, depth: raw.depth || 0,
      appear: 0, hoverAmt: 0, hidden: false, chipW: 120, chipH: 32, text: raw.label
    }));
    state.nodes = nodes;
    state.byId = new Map(nodes.map((node) => [node.id, node]));
    state.links = RAW.links.map((link) => Object.assign({}, link, {
      a: state.byId.get(link.source),
      b: state.byId.get(link.target)
    })).filter((link) => link.a && link.b);

    for (const node of nodes) sizeNode(node);
    loadLayout();
    relayout();
    for (const node of nodes) {
      node.x = (node.tx || 0) * 0.92;
      node.y = (node.ty || 0) * 0.92;
      node.appear = 0;
    }
    fit();
    state.view.k = Math.max(0.28, state.target.k * 0.82);
    state.view.x = state.target.x;
    state.view.y = state.target.y;

    const kinds = [...new Set(nodes.filter((node) => node.type === "record").map((node) => node.kind || "untyped"))].sort();
    kindSelect.innerHTML = '<option value="">all kinds</option>' + kinds.map((kind) =>
      '<option value="' + esc(kind) + '">' + esc(kind) + "</option>").join("");

    renderDetail();
    window.__lethix = state;
    state._render = render;
    state._frame = frame;
    requestAnimationFrame(frame);
  }

  new ResizeObserver(() => resize()).observe(stage);
  boot();
})();
"""

_PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__STYLE__</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>UM-MEM memory graph</h1>
    <div class="sub">__SUBTITLE__</div>
  </div>
  <div class="tools">
    <input id="q" type="search" placeholder="Find a record or tag" autocomplete="off">
    <select id="kind"></select>
    <label class="chip-toggle"><input id="links" type="checkbox" checked> links</label>
    <button id="theme" type="button" title="Dark theme">☾</button>
    <button id="fit" type="button">fit</button>
  </div>
  <div class="stats">__STATS__</div>
</header>
<main>
  <div id="stage">
    <canvas id="graph"></canvas>
    <div id="tooltip"></div>
    <div id="matches" class="matches"></div>
    <div class="hint">drag a card to move it · drag empty space to pan · scroll to zoom · double-click to reset</div>
  </div>
  <aside id="detail"></aside>
</main>
<script id="map-data" type="application/json">__DATA__</script>
<script>__SCRIPT__</script>
</body>
</html>
"""


def _escape(value: object) -> str:
    return html.escape(str(value or ""))


def _age(recorded_at: str) -> str:
    raw = str(recorded_at or "").strip()
    if not raw:
        return "unknown age"
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return "unknown age"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    seconds = int((datetime.now(UTC) - moment).total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{max(1, seconds // 60)} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def _collect_tree(
    nodes: Iterable[Mapping[str, Any]], parents: Sequence[str] = ()
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, Mapping):
            continue
        name = str(node.get("name") or "")
        path_names = [*parents, name] if name else list(parents)
        collected.append(
            {
                "block_id": str(node.get("block_id") or ""),
                "name": name,
                "description": str(node.get("description") or ""),
                "synopsis": str(node.get("synopsis") or ""),
                "tags": [str(tag) for tag in node.get("tags") or ()],
                "records": int(node.get("canonical_record_count") or 0),
                "writable": bool(node.get("writable")),
                "index_only": bool(node.get("index_only")),
                "meta": bool(node.get("meta")),
                "node_kind": str(node.get("node_kind") or ""),
                "path_names": path_names,
                "children": _collect_tree(node.get("children") or [], path_names),
            }
        )
    return collected


def _leaves(tree: Sequence[Mapping[str, Any]]) -> list[str]:
    found: list[str] = []

    def walk(nodes: Sequence[Mapping[str, Any]]) -> None:
        for node in nodes:
            children = node.get("children") or []
            if children:
                walk(children)
            elif str(node.get("block_id") or ""):
                found.append(str(node["block_id"]))

    walk(tree)
    return found


def _collect_records(
    client: MemoryClient, tree: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    from um_mem.memory import procedures as procedure_rules

    records: list[dict[str, Any]] = []
    for block_id in _leaves(tree):
        cursor = ""
        while True:
            page = client.list_records(
                scope="workspace",
                scope_id=client.workspace_id,
                logical_block_id=block_id,
                limit=128,
                cursor=cursor,
            )
            identifiers = [
                str(item.get("record_id") or "")
                for item in page.get("records") or []
                if str(item.get("record_id") or "")
            ]
            for start in range(0, len(identifiers), 128):
                for record in client.read_exact_records(identifiers[start : start + 128]):
                    metadata = getattr(record, "metadata", {}) or {}
                    steps = metadata.get("steps")
                    freshness = ""
                    if str(getattr(record, "kind", "") or "") == "procedure":
                        freshness = procedure_rules.procedure_freshness(
                            metadata if isinstance(metadata, Mapping) else {}
                        )["freshness"]
                    records.append(
                        {
                            "record_id": str(getattr(record, "block_id", "") or ""),
                            "title": str(getattr(record, "title", "") or ""),
                            "content": str(getattr(record, "content", "") or ""),
                            "kind": str(getattr(record, "kind", "") or ""),
                            "tags": [str(tag) for tag in (getattr(record, "tags", ()) or ())],
                            "status": str(getattr(record, "status", "") or ""),
                            "recorded_at": str(getattr(record, "recorded_at", "") or ""),
                            "block": block_id,
                            "steps": [str(step) for step in steps]
                            if isinstance(steps, (list, tuple))
                            else [],
                            "verify": str(metadata.get("verify") or "")
                            if isinstance(metadata, Mapping)
                            else "",
                            "freshness": freshness,
                        }
                    )
            cursor = str(page.get("next_cursor") or "")
            if not cursor:
                break
    return records


def _semantic_edges(
    records: Sequence[Mapping[str, Any]],
    repo_root: Path,
    workspace_id: str,
    profile: str,
) -> list[dict[str, Any]]:
    """Return the record pairs that mean something to each other."""

    from um_mem.memory import semantic as semantic_tier

    embedder = semantic_tier.get_embedder(cache_dir=semantic_tier.default_cache_dir(repo_root))
    if embedder is None or len(records) < 2:
        return []
    index = semantic_tier.SemanticIndex(semantic_tier.semantic_index_path(repo_root, workspace_id))
    vectors = semantic_tier.cached_vectors(index, embedder.model, profile)
    if not vectors:
        return []
    floor = semantic_tier.calibrated_floor(repo_root, workspace_id, embedder.model)
    threshold = max(SEMANTIC_LINK_FLOOR, floor or 0.0)
    pairs: list[tuple[float, str, str]] = []
    identifiers = [str(record["record_id"]) for record in records]
    for position, first in enumerate(identifiers):
        left = vectors.get(first)
        if not left:
            continue
        for second in identifiers[position + 1 :]:
            right = vectors.get(second)
            if not right:
                continue
            score = semantic_tier._cosine(
                left.get("full", ("", []))[1], right.get("full", ("", []))[1]
            )
            if score >= threshold:
                pairs.append((float(score), first, second))
    pairs.sort(reverse=True)
    degree: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    for score, first, second in pairs:
        draw = (
            degree.get(first, 0) < SEMANTIC_LINKS_PER_RECORD
            and degree.get(second, 0) < SEMANTIC_LINKS_PER_RECORD
        )
        if draw:
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
        edges.append(
            {
                "source": first,
                "target": second,
                "kind": "semantic",
                "weight": round(score, 4),
                "draw": draw,
            }
        )
    return edges


def collect_memory_map(
    repo_root: Path | str,
    workspace_id: str = "",
    profile_id: str = "default",
    backend: str = "",
) -> dict[str, Any]:
    """Collect everything the graph shows."""

    from um_mem.memory import semantic as semantic_tier

    client = MemoryClient(
        Path(repo_root),
        workspace_id=workspace_id,
        profile_id=profile_id,
        backend=backend,
        session_id="memory-map",
    )
    try:
        roots = client.block_map(depth=5, include_meta=True)
        tree = _collect_tree(roots)
        records = _collect_records(client, tree)
        indexed = 0
        model = ""
        try:
            embedder = semantic_tier.get_embedder(
                cache_dir=semantic_tier.default_cache_dir(client.repo_root)
            )
            if embedder is not None:
                model = embedder.model
                index = semantic_tier.SemanticIndex(
                    semantic_tier.semantic_index_path(client.repo_root, client.workspace_id)
                )
                indexed = index.count(embedder.model, profile_id)
        except Exception:  # pragma: no cover - optional tier
            model = ""
        edges = _semantic_edges(records, client.repo_root, client.workspace_id, profile_id)
        kinds: dict[str, int] = {}
        for record in records:
            kinds[record["kind"] or "untyped"] = kinds.get(record["kind"] or "untyped", 0) + 1
        principles = sum(1 for node in _walk_blocks(tree) if node.get("meta"))
        return {
            "workspace_id": client.workspace_id,
            "profile_id": profile_id,
            "backend": str(getattr(client, "backend_id", "") or backend or ""),
            "tree": tree,
            "records": records,
            "record_count": len(records),
            "block_count": sum(1 for _ in _walk_blocks(tree)),
            "meta_block_count": principles,
            "indexed_records": indexed,
            "semantic_model": model,
            "kinds": kinds,
            "edges": edges,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _walk_blocks(tree: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for node in tree or []:
        yield node
        yield from _walk_blocks(node.get("children") or [])


def _graph_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the collected map into graph nodes and links."""

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    def walk(items: Sequence[Mapping[str, Any]], parent: str, depth: int) -> None:
        for node in items:
            block_id = str(node.get("block_id") or "")
            if not block_id:
                continue
            nodes.append(
                {
                    "id": block_id,
                    "type": "block",
                    "label": str(node.get("name") or ""),
                    "description": str(node.get("description") or ""),
                    "synopsis": str(node.get("synopsis") or ""),
                    "tags": list(node.get("tags") or []),
                    "writable": bool(node.get("writable")),
                    "index_only": bool(node.get("index_only")),
                    "meta": bool(node.get("meta")),
                    "node_kind": str(node.get("node_kind") or ""),
                    "records": int(node.get("records") or 0),
                    "parent": parent,
                    "depth": depth,
                    "path_names": list(node.get("path_names") or []),
                }
            )
            if parent:
                links.append({"source": parent, "target": block_id, "kind": "tree"})
            walk(node.get("children") or [], block_id, depth + 1)

    walk(data.get("tree") or [], "", 0)

    for record in data.get("records") or []:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            continue
        haystack = " ".join(
            [
                str(record.get("title") or ""),
                str(record.get("content") or ""),
                " ".join(record.get("tags") or []),
                str(record.get("kind") or ""),
            ]
        ).lower()
        nodes.append(
            {
                "id": record_id,
                "type": "record",
                "label": str(record.get("title") or "(untitled)"),
                "kind": str(record.get("kind") or ""),
                "text": str(record.get("content") or ""),
                "tags": list(record.get("tags") or []),
                "status": str(record.get("status") or ""),
                "age": _age(str(record.get("recorded_at") or "")),
                "steps": list(record.get("steps") or []),
                "verify": str(record.get("verify") or ""),
                "freshness": str(record.get("freshness") or ""),
                "parent": str(record.get("block") or ""),
                "hay": haystack,
            }
        )
        if record.get("block"):
            links.append(
                {
                    "source": str(record["block"]),
                    "target": record_id,
                    "kind": "membership",
                }
            )

    identifiers = {node["id"] for node in nodes}
    for edge in data.get("edges") or []:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in identifiers or target not in identifiers:
            continue
        links.append(
            {
                "source": source,
                "target": target,
                "kind": "semantic",
                "weight": edge.get("weight"),
                "draw": bool(edge.get("draw", True)),
            }
        )
    return {"nodes": nodes, "links": links}


def render_memory_map(data: Mapping[str, Any]) -> str:
    """Return the graph as one self-contained HTML document."""

    graph = _graph_data(data)
    sorted({str(node.get("kind") or "") for node in graph["nodes"] if node["type"] == "record"})
    record_count = int(data.get("record_count") or 0)
    indexed = int(data.get("indexed_records") or 0)
    drawn_links = sum(
        1 for link in graph["links"] if link.get("kind") == "semantic" and link.get("draw")
    )
    block_count = int(data.get("block_count") or 0)
    meta_count = int(data.get("meta_block_count") or 0)
    payload = json.dumps(graph, ensure_ascii=False).replace("</", "<\\/")
    full_model = str(data.get("semantic_model") or "off")
    short_model = full_model.rsplit("/", 1)[-1]
    real_blocks = max(0, block_count - meta_count)
    stats = (
        f'<div class="stat"><b>{record_count}</b>records</div>'
        f'<div class="stat"><b>{real_blocks}</b>blocks</div>'
        f'<div class="stat"><b>{drawn_links}</b>links</div>'
        f'<div class="stat" title="{_escape(full_model)}">'
        f"<b>{_escape(short_model)}</b>model</div>"
    )
    subtitle = " · ".join(
        part
        for part in (
            str(data.get("workspace_id") or ""),
            str(data.get("backend") or ""),
            f"{indexed}/{record_count} indexed" if record_count else "",
            f"{meta_count} meta" if meta_count else "",
        )
        if part
    )
    return (
        _PAGE_HEAD.replace(
            "__TITLE__", f"UM-MEM memory graph · {_escape(data.get('workspace_id'))}"
        )
        .replace("__SUBTITLE__", _escape(subtitle))
        .replace("__STATS__", stats)
        .replace("__STYLE__", _STYLE)
        .replace("__DATA__", payload)
        .replace("__SCRIPT__", _SCRIPT)
    )


def write_memory_map(
    repo_root: Path | str,
    output_path: Path | str,
    workspace_id: str = "",
    profile_id: str = "default",
    backend: str = "",
) -> Path:
    """Write the memory graph to a file and return its path."""

    data = collect_memory_map(
        repo_root, workspace_id=workspace_id, profile_id=profile_id, backend=backend
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_memory_map(data), encoding="utf-8")
    return target


def memory_map_summary(data: Mapping[str, Any]) -> str:
    """Return a one-line summary of a collected map."""

    return json.dumps(
        {
            "workspace_id": data.get("workspace_id"),
            "records": data.get("record_count"),
            "blocks": data.get("block_count"),
            "meta_blocks": data.get("meta_block_count"),
            "indexed": data.get("indexed_records"),
            "model": data.get("semantic_model"),
            "kinds": data.get("kinds"),
            "semantic_links": sum(1 for edge in data.get("edges") or [] if edge.get("draw", True)),
        },
        ensure_ascii=False,
    )
