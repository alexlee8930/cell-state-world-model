import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";
import {
  interpret,
  runPlan,
  intervene as apiIntervene,
  ask,
  health,
  type Frames,
  type Plan,
} from "@/api/worldModelApi";
import { ImageViewport, type Annotation } from "@/components/ImageViewport";

// Validation numbers from the real Xenium run (shown, never recomputed here).
const PROV = { brightnessR: 0.68, sizeR: 0.66, shuffleR: -0.09 };

let _n = 0;
const uid = (p = "id") => `${p}${++_n}`;
const oidStr = (n: number) => `cell_${String(n).padStart(4, "0")}`;

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const s = localStorage.getItem("cwm.theme");
    return s === "light" || s === "dark" ? s : "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("cwm.theme", theme);
  }, [theme]);
  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

type LogKind = "cmd" | "ok" | "warn";
interface LogLine { id: string; kind: LogKind; text: string }
interface Intervention { id: string; at: number; toward: number; strength: number }
interface CellObject {
  id: string; oid: string; label: string;
  plan: Plan; frames: Frames;
  annotations: Annotation[];
  interventions: Intervention[];
}
interface Session { id: string; name: string; log: LogLine[]; objects: CellObject[]; selectedId: string | null }

const EXAMPLES = [
  "an invasive mesenchymal cell",
  "an epithelial cell that undergoes EMT over time",
  "a proliferative crypt cell, rolled forward over time",
  "a cell perturbed toward an immune phenotype halfway through",
];

export default function App() {
  const { theme, toggle } = useTheme();
  const [sessions, setSessions] = useState<Session[]>(() => [
    { id: uid("ses"), name: "Untitled", log: [], objects: [], selectedId: null },
  ]);
  const [activeId, setActiveId] = useState(sessions[0].id);
  const [input, setInput] = useState("");
  const [tab, setTab] = useState<"state" | "annot" | "intervene">("state");
  const [busy, setBusy] = useState(false);
  const [replying, setReplying] = useState(false);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  // intervention controls
  const [ivAt, setIvAt] = useState(3);
  const [ivToward, setIvToward] = useState(9000);
  const [ivStrength, setIvStrength] = useState(0.9);
  const logRef = useRef<HTMLDivElement>(null);
  const oidRef = useRef(1);
  const timer = useRef<number | null>(null);

  const active = sessions.find((s) => s.id === activeId)!;
  const obj = active.objects.find((o) => o.id === active.selectedId) ?? active.objects.at(-1) ?? null;
  const n = obj?.frames.images.length ?? 0;
  const cur = obj?.frames.images[Math.min(t, Math.max(0, n - 1))];
  const post = obj?.frames.interveneAt != null && t >= obj.frames.interveneAt;
  const dist = obj?.frames.dist?.[t];

  useEffect(() => { health().then((h) => setOnline(!!h.ok && h.hasKey)).catch(() => setOnline(false)); }, []);
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight }); }, [active.log]);
  useEffect(() => { setT(0); setPlaying(false); }, [active.selectedId]);
  useEffect(() => {
    if (playing && n > 1) {
      timer.current = window.setInterval(() => setT((x) => (x + 1) % n), 500);
      return () => { if (timer.current) window.clearInterval(timer.current); };
    }
  }, [playing, n]);

  const patch = (fn: (s: Session) => Session) => setSessions((ss) => ss.map((s) => (s.id === activeId ? fn(s) : s)));
  const patchObj = (id: string, fn: (o: CellObject) => CellObject) =>
    patch((s) => ({ ...s, objects: s.objects.map((o) => (o.id === id ? fn(o) : o)) }));
  const log = (kind: LogKind, text: string) =>
    patch((s) => ({ ...s, log: [...s.log, { id: uid("l"), kind, text }] }));

  // ── console: state text → Claude plan → model generation ──────────────────
  const run = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    patch((s) => ({ ...s, name: s.log.length === 0 ? text.slice(0, 26) : s.name,
      log: [...s.log, { id: uid("l"), kind: "cmd" as LogKind, text }] }));
    try {
      const plan = await interpret(text);
      log("ok", `plan: ${plan.mode} cell=${plan.cellId}${plan.end != null ? `→${plan.end}` : ""} steps=${plan.steps ?? 1} w=${plan.w ?? 3}${plan.rationale ? " · " + plan.rationale : ""}`);
      const frames = await runPlan(plan);
      const o: CellObject = {
        id: uid("obj"), oid: oidStr(oidRef.current++), label: text, plan, frames,
        annotations: [], interventions: [],
      };
      if (plan.intervene) { setIvAt(plan.intervene.at); setIvToward(plan.intervene.toward); setIvStrength(plan.intervene.strength); }
      patch((s) => ({ ...s, objects: [...s.objects, o], selectedId: o.id,
        log: [...s.log, { id: uid("l"), kind: "ok" as LogKind, text: `${o.oid} generated · ${frames.info ?? ""}` }] }));
      setT(0);
    } catch (e: any) {
      log("warn", `${e.message || e}`);
    } finally { setBusy(false); }
  };

  // ── intervention: push the state mid-trajectory and re-generate ───────────
  const applyIntervene = async () => {
    if (!obj || busy) return;
    setBusy(true); setPlaying(false);
    log("cmd", `intervene at t=${ivAt} toward=${ivToward} strength=${ivStrength}`);
    try {
      const p = obj.plan;
      const f = await apiIntervene(p.cellId, p.end ?? p.cellId + 1000, p.steps ?? Math.max(n, 6),
        ivAt, ivToward, ivStrength, p.w ?? 3);
      patchObj(obj.id, (o) => ({
        ...o, frames: f,
        interventions: [...o.interventions, { id: uid("iv"), at: ivAt, toward: ivToward, strength: ivStrength }],
      }));
      setT(0);
      log("ok", f.info ?? "intervened");
    } catch (e: any) {
      log("warn", `${e.message || e}`);
    } finally { setBusy(false); }
  };

  // ── annotations (threads on the real generated image, answered by Claude) ─
  const addAnnotation = (x: number, y: number, text: string) =>
    obj && patchObj(obj.id, (o) => ({ ...o, annotations: [...o.annotations, { id: uid("an"), x, y, comments: [{ author: "you", text }] }] }));
  const removeAnnotation = (id: string) =>
    obj && patchObj(obj.id, (o) => ({ ...o, annotations: o.annotations.filter((a) => a.id !== id) }));
  const reply = async (annId: string, text: string) => {
    if (!obj) return;
    patchObj(obj.id, (o) => ({ ...o, annotations: o.annotations.map((a) =>
      a.id === annId ? { ...a, comments: [...a.comments, { author: "you" as const, text }] } : a) }));
    const a = obj.annotations.find((x) => x.id === annId);
    setReplying(true);
    try {
      const answer = await ask(text, {
        object: obj.oid, mode: obj.plan.mode, cellId: obj.plan.cellId, end: obj.plan.end,
        guidance_w: obj.plan.w ?? 3, frames: n, frame_index: t,
        S_distance_from_start: dist, S_norm: obj.frames.sNorm,
        clicked_at: a ? { x: +a.x.toFixed(3), y: +a.y.toFixed(3) } : undefined,
        validation: PROV,
      });
      patchObj(obj.id, (o) => ({ ...o, annotations: o.annotations.map((x) =>
        x.id === annId ? { ...x, comments: [...x.comments, { author: "model" as const, text: answer }] } : x) }));
    } catch (e: any) {
      patchObj(obj.id, (o) => ({ ...o, annotations: o.annotations.map((x) =>
        x.id === annId ? { ...x, comments: [...x.comments, { author: "model" as const, text: `(error: ${e.message || e})` }] } : x) }));
    } finally { setReplying(false); }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-bg text-text">
      {/* menu bar */}
      <div className="flex items-center justify-between border-b border-border bg-surface px-3 py-1.5">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[12px] font-semibold tracking-tight text-text">CellState Workbench</span>
          <span className="font-mono text-[10px] text-muted">v0.2</span>
          <span className="border-l border-border pl-3 font-mono text-[10px] text-muted">specimen: Xenium CRC · 422-plex · z=7</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px]" style={{ color: online === false ? "var(--error)" : "var(--muted)" }}>
            {online == null ? "connecting…" : online ? "encoder E + diffusion decoder · online" : "proxy offline (npm run proxy)"}
          </span>
          <button onClick={toggle} className="border border-border px-2 py-0.5 font-mono text-[10px] text-muted hover:bg-surface-2">
            {theme === "dark" ? "light" : "dark"}
          </button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* left: sessions */}
        <aside className="flex w-52 shrink-0 flex-col border-r border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border px-2 py-1">
            <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-muted">Sessions</span>
            <button onClick={() => {
              const s: Session = { id: uid("ses"), name: "Untitled", log: [], objects: [], selectedId: null };
              setSessions((ss) => [s, ...ss]); setActiveId(s.id);
            }} className="border border-border px-1.5 text-[12px] leading-none text-muted hover:bg-surface-2">+</button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {sessions.map((s) => (
              <button key={s.id} onClick={() => setActiveId(s.id)}
                className={cn("block w-full border-b border-border-faint px-2 py-1.5 text-left",
                  s.id === activeId ? "bg-accent/10" : "hover:bg-surface-2")}>
                <div className="truncate text-[12px] text-text">{s.name}</div>
                <div className="font-mono text-[10px] text-muted">{s.objects.length} object(s)</div>
              </button>
            ))}
          </div>
          <div className="border-t border-border p-2 font-mono text-[9px] leading-relaxed text-muted">
            expr→morph validation<br />
            brightness r={PROV.brightnessR} · size r={PROV.sizeR}<br />
            shuffle r={PROV.shuffleR}
          </div>
        </aside>

        {/* center: viewport + console */}
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-3 border-b border-border bg-surface px-3 py-1">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted">Viewport</span>
            {obj && (
              <>
                <span className="font-mono text-[10px] text-muted">object: {obj.oid}</span>
                <span className="font-mono text-[10px] text-muted">mode: {obj.plan.mode}</span>
                <span className="font-mono text-[10px] text-muted">w: {obj.plan.w ?? 3}</span>
              </>
            )}
          </div>

          <div className="grid min-h-0 flex-1 place-items-center overflow-auto p-4"
            style={{ backgroundImage: "radial-gradient(var(--border-faint) 1px, transparent 1px)", backgroundSize: "16px 16px" }}>
            {cur && obj ? (
              <div className="flex flex-col items-center gap-3">
                <div className="flex items-end gap-4">
                  {obj.frames.real && (
                    <figure className="flex flex-col items-center gap-1">
                      <img src={obj.frames.real} width={88} height={88} alt="real start"
                        style={{ imageRendering: "pixelated", width: 88 }} className="border border-border bg-black" />
                      <figcaption className="font-mono text-[9px] uppercase text-muted">real · start</figcaption>
                    </figure>
                  )}
                  <ImageViewport
                    key={obj.id}
                    src={cur}
                    size={300}
                    annotations={obj.annotations}
                    post={!!post}
                    replying={replying}
                    channel={post ? "CH: generated · post-intervention" : "CH: generated from state S"}
                    onAddAnnotation={addAnnotation}
                    onRemoveAnnotation={removeAnnotation}
                    onReply={reply}
                  />
                  {(obj.frames.realToward ?? obj.frames.realEnd) && (
                    <figure className="flex flex-col items-center gap-1">
                      <img src={obj.frames.realToward ?? obj.frames.realEnd} width={88} height={88} alt="real target"
                        style={{ imageRendering: "pixelated", width: 88 }} className="border border-border bg-black" />
                      <figcaption className="font-mono text-[9px] uppercase text-muted">
                        {obj.frames.realToward ? "real · target" : "real · end"}
                      </figcaption>
                    </figure>
                  )}
                </div>

                {n > 1 && (
                  <div className="flex w-[420px] items-center gap-2">
                    <button onClick={() => setPlaying((p) => !p)}
                      className="border border-border px-2 py-0.5 font-mono text-[11px] text-text hover:bg-surface-2">
                      {playing ? "❚❚" : "▶"}
                    </button>
                    <input type="range" min={0} max={n - 1} step={1} value={t}
                      onChange={(e) => { setPlaying(false); setT(+e.target.value); }} className="flex-1" />
                    <span className="w-12 text-right font-mono text-[10px] text-muted">t {t}/{n - 1}</span>
                    {dist != null && <span className="w-20 text-right font-mono text-[10px] text-text">ΔS {dist.toFixed(1)}</span>}
                  </div>
                )}
              </div>
            ) : (
              <div className="font-mono text-[12px] text-muted">
                {busy ? "generating on GPU…" : "no object — describe a cell state in the console below"}
              </div>
            )}
          </div>

          {/* objects strip */}
          {active.objects.length > 0 && (
            <div className="flex items-center gap-2 border-t border-border bg-surface px-3 py-1.5">
              <span className="font-mono text-[10px] uppercase tracking-wider text-muted">Objects</span>
              <div className="flex gap-1.5 overflow-x-auto">
                {active.objects.map((o) => (
                  <button key={o.id} onClick={() => patch((s) => ({ ...s, selectedId: o.id }))}
                    title={`${o.oid} · ${o.label}`}
                    className={cn("shrink-0 border", o.id === obj?.id ? "border-accent" : "border-border hover:border-muted")}>
                    <img src={o.frames.images[0]} width={44} height={44} alt={o.oid}
                      style={{ imageRendering: "pixelated", width: 44, display: "block" }} />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* console */}
          <div className="flex h-40 shrink-0 flex-col border-t border-border bg-surface">
            <div className="border-b border-border px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-muted">Console</div>
            <div ref={logRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-1.5 font-mono text-[11px] leading-relaxed">
              {active.log.length === 0 && (
                <div className="text-muted">// describe a cell state — Claude plans it, the model generates it</div>
              )}
              {active.log.map((l) => (
                <div key={l.id} className={cn(l.kind === "cmd" ? "text-text" : l.kind === "ok" ? "text-ok" : "text-warn")}>
                  <span className="text-muted">{l.kind === "cmd" ? "> " : l.kind === "ok" ? "[ok]  " : "[warn]"}</span>
                  {l.text}
                </div>
              ))}
              {busy && <div className="text-muted">… working (GPU cold start can take ~30–60s)</div>}
            </div>
            <div className="flex flex-wrap gap-1 px-3 py-1">
              {EXAMPLES.map((ex) => (
                <button key={ex} onClick={() => run(ex)} disabled={busy}
                  className="border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted hover:border-accent hover:text-accent disabled:opacity-40">
                  {ex}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1 border-t border-border px-3 py-1.5">
              <span className="font-mono text-[12px] text-accent">state&gt;</span>
              <input value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && run()}
                placeholder="describe a cell state — e.g. an epithelial cell that undergoes EMT over time"
                className="min-w-0 flex-1 bg-transparent font-mono text-[12px] outline-none placeholder:text-muted" />
              <button onClick={() => run()} disabled={busy || !input.trim()}
                className="border border-accent bg-accent px-2 py-0.5 font-mono text-[10px] font-semibold uppercase text-accent-fg disabled:opacity-30">
                run
              </button>
            </div>
          </div>
        </main>

        {/* right: inspector */}
        <aside className="flex w-[340px] shrink-0 flex-col border-l border-border bg-surface">
          <div className="flex border-b border-border">
            {(["state", "annot", "intervene"] as const).map((x) => (
              <button key={x} onClick={() => setTab(x)}
                className={cn("flex-1 border-r border-border py-1.5 font-mono text-[10px] uppercase tracking-wider last:border-r-0",
                  tab === x ? "bg-accent/10 text-text" : "text-muted hover:bg-surface-2")}>
                {x === "state" ? "State" : x === "annot" ? `Annot (${obj?.annotations.length ?? 0})` : `Intervene (${obj?.interventions.length ?? 0})`}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {!obj ? (
              <div className="p-3 font-mono text-[11px] text-muted">no object selected</div>
            ) : tab === "state" ? (
              <div className="p-2">
                <table className="w-full font-mono text-[11px]">
                  <tbody>
                    <tr className="text-muted"><td colSpan={2} className="pb-1 text-[9px] uppercase tracking-wider">plan (from Claude)</td></tr>
                    {([["mode", obj.plan.mode], ["cell id", obj.plan.cellId], ["end", obj.plan.end ?? "—"],
                       ["steps", obj.plan.steps ?? 1], ["guidance w", obj.plan.w ?? 3]] as const).map(([k, v]) => (
                      <tr key={k} className="border-t border-border-faint">
                        <td className="py-0.5 pr-2 text-muted">{k}</td>
                        <td className="py-0.5 text-right text-text">{String(v)}</td>
                      </tr>
                    ))}
                    <tr className="text-muted"><td colSpan={2} className="pb-1 pt-3 text-[9px] uppercase tracking-wider">state S (live)</td></tr>
                    {([["frames", n], ["frame index", t],
                       ["ΔS from start", dist != null ? dist.toFixed(2) : "—"],
                       ["|S| (norm)", obj.frames.sNorm != null ? obj.frames.sNorm.toFixed(1) : "—"]] as const).map(([k, v]) => (
                      <tr key={k} className="border-t border-border-faint">
                        <td className="py-0.5 pr-2 text-muted">{k}</td>
                        <td className="py-0.5 text-right text-text">{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {obj.plan.rationale && (
                  <div className="mt-3 border border-border-faint bg-surface-2 p-2 font-mono text-[9px] leading-relaxed text-muted">
                    {obj.plan.rationale}
                  </div>
                )}
                <div className="mt-2 border border-border-faint bg-surface-2 p-2 font-mono text-[9px] leading-relaxed text-muted">
                  Images are generated from the learned state S by a CFG diffusion decoder (not procedural).
                  Size/brightness are validated signal (r≈{PROV.sizeR}–{PROV.brightnessR}); fine texture is illustrative.
                </div>
              </div>
            ) : tab === "annot" ? (
              <table className="w-full font-mono text-[11px]">
                <thead className="text-muted">
                  <tr className="border-b border-border">
                    <th className="px-2 py-1 text-left font-normal">#</th>
                    <th className="px-2 py-1 text-left font-normal">x</th>
                    <th className="px-2 py-1 text-left font-normal">y</th>
                    <th className="px-2 py-1 text-left font-normal">thread</th>
                  </tr>
                </thead>
                <tbody>
                  {obj.annotations.length === 0 && (
                    <tr><td colSpan={4} className="px-2 py-2 text-muted">no annotations — click the viewport</td></tr>
                  )}
                  {obj.annotations.map((a, i) => (
                    <tr key={a.id} className="border-b border-border-faint">
                      <td className="px-2 py-1 text-accent">{i + 1}</td>
                      <td className="px-2 py-1 text-text">{a.x.toFixed(3)}</td>
                      <td className="px-2 py-1 text-text">{a.y.toFixed(3)}</td>
                      <td className="px-2 py-1 text-muted">{a.comments.length} · {a.comments[0].text.slice(0, 20)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-2">
                <div className="mb-2 font-mono text-[9px] uppercase tracking-wider text-muted">Push the state mid-trajectory</div>
                <div className="space-y-2 font-mono text-[11px]">
                  <label className="flex items-center gap-2">
                    <span className="w-16 text-muted">at t</span>
                    <input type="range" min={0} max={Math.max(1, n - 1)} step={1} value={ivAt}
                      onChange={(e) => setIvAt(+e.target.value)} className="flex-1" />
                    <span className="w-6 text-right text-text">{ivAt}</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <span className="w-16 text-muted">toward id</span>
                    <input type="number" value={ivToward} onChange={(e) => setIvToward(+e.target.value)}
                      className="flex-1 border border-border bg-surface-2 px-1 py-0.5 text-text outline-none" />
                  </label>
                  <label className="flex items-center gap-2">
                    <span className="w-16 text-muted">strength</span>
                    <input type="range" min={0} max={1} step={0.05} value={ivStrength}
                      onChange={(e) => setIvStrength(+e.target.value)} className="flex-1" />
                    <span className="w-8 text-right text-text">{ivStrength.toFixed(2)}</span>
                  </label>
                  <button onClick={applyIntervene} disabled={busy || n < 2}
                    className="w-full border py-1 text-[11px] font-semibold uppercase tracking-wide text-text disabled:opacity-40"
                    style={{ borderColor: "var(--series-5)" }}>
                    {busy ? "generating…" : "apply intervention"}
                  </button>
                  {n < 2 && <div className="text-[10px] text-muted">needs a trajectory (ask for a state "over time")</div>}
                </div>

                <div className="mt-3 font-mono text-[9px] uppercase tracking-wider text-muted">
                  history — each apply replaces the frames; the last row is in effect
                </div>
                <table className="w-full font-mono text-[11px]">
                  <thead className="text-muted">
                    <tr className="border-b border-border">
                      <th className="px-1 py-1 text-left font-normal">at t</th>
                      <th className="px-1 py-1 text-left font-normal">toward</th>
                      <th className="px-1 py-1 text-left font-normal">strength</th>
                      <th className="px-1 py-1 text-right font-normal"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {obj.interventions.length === 0 && (
                      <tr><td colSpan={4} className="px-1 py-2 text-muted">none applied</td></tr>
                    )}
                    {obj.interventions.map((iv, i) => {
                      const activeRow = i === obj.interventions.length - 1;
                      return (
                        <tr key={iv.id} className={cn("border-b border-border-faint", !activeRow && "opacity-50")}>
                          <td className="px-1 py-1 text-text">{iv.at}</td>
                          <td className="px-1 py-1 text-text">{iv.toward}</td>
                          <td className="px-1 py-1 text-text">{iv.strength.toFixed(2)}</td>
                          <td className="px-1 py-1 text-right text-[9px] uppercase"
                            style={activeRow ? { color: "var(--series-5)" } : undefined}>
                            {activeRow ? "active" : "superseded"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* status bar */}
      <div className="flex items-center justify-between border-t border-border bg-surface px-3 py-1 font-mono text-[10px] text-muted">
        <div className="flex gap-4">
          <span>session: {active.name}</span>
          <span>object: {obj?.oid ?? "—"}</span>
          <span>mode: {obj?.plan.mode ?? "—"}</span>
        </div>
        <div className="flex gap-4">
          <span>frames: {n || "—"}</span>
          <span>annot: {obj?.annotations.length ?? 0}</span>
          <span>interv: {obj?.interventions.length ?? 0}</span>
          <span className="text-ok">generated from state S · CFG diffusion</span>
        </div>
      </div>
    </div>
  );
}
