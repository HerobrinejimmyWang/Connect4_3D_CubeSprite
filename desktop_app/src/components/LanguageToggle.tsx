import type { Language } from "../types";

interface Props {
  language: Language;
  onChange: (language: Language) => void;
}

export function LanguageToggle({ language, onChange }: Props) {
  return (
    <div className="language-toggle" role="group" aria-label="Language / 语言">
      <button className={language === "zh" ? "active" : ""} onClick={() => onChange("zh")} aria-pressed={language === "zh"}>
        中
      </button>
      <button className={language === "en" ? "active" : ""} onClick={() => onChange("en")} aria-pressed={language === "en"}>
        EN
      </button>
    </div>
  );
}
