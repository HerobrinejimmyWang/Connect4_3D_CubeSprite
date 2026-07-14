import type { ReactNode } from "react";

import type { Copy } from "../i18n";

interface Props {
  copy: Copy;
  title: string;
  subtitle?: string;
  children: ReactNode;
  onBack: () => void;
  wide?: boolean;
}

export function PageShell({ copy, title, subtitle, children, onBack, wide = false }: Props) {
  return (
    <main className={`page-shell ${wide ? "wide" : ""}`}>
      <header className="page-heading">
        <div className="eyebrow">CUBESPRITE · 0.1.0</div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </header>
      <section className="page-content">{children}</section>
      <button className="back-button" onClick={onBack}>
        <span aria-hidden="true">←</span> {copy.common.back}
      </button>
    </main>
  );
}
