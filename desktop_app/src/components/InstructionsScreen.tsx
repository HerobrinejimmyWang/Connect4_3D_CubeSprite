import type { Copy } from "../i18n";
import { PageShell } from "./PageShell";

interface Props { copy: Copy; onBack: () => void }

export function InstructionsScreen({ copy: t, onBack }: Props) {
  const sections = [
    ["01", t.instructions.rulesTitle, t.instructions.rules],
    ["02", t.instructions.controlsTitle, t.instructions.controls],
    ["03", t.instructions.quickTitle, t.instructions.quick],
  ] as const;
  return (
    <PageShell copy={t} title={t.instructions.title} subtitle={t.instructions.subtitle} onBack={onBack} wide>
      <div className="instruction-grid">
        {sections.map(([number, title, items]) => (
          <article className="instruction-card" key={number}>
            <header><span>{number}</span><h2>{title}</h2></header>
            <ol>{items.map((item) => <li key={item}>{item}</li>)}</ol>
          </article>
        ))}
      </div>
    </PageShell>
  );
}
