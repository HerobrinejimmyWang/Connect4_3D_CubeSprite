import type { ReactNode } from "react";

import type { Copy } from "../i18n";

interface Props {
  copy: Copy;
  title: string;
  subtitle?: string;
  children: ReactNode;
  onBack: () => void;
  wide?: boolean;
  backLabel?: string;
}

export function PageShell({ copy, title, subtitle, children, onBack, wide = false, backLabel }: Props) {
  return (
    <main className={`page-shell ${wide ? "wide" : ""}`}>
      <header className="page-heading">
        <div className="eyebrow">CUBESPRITE</div>
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </header>
      <section className="page-content">{children}</section>
      <button className="back-button" onClick={onBack}>
        <span aria-hidden="true">←</span> {backLabel ?? copy.common.back}
      </button>
    </main>
  );
}
