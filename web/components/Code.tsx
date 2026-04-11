type Props = {
  code: string;
  language?: string;
};

export default function Code({ code, language = "python" }: Props) {
  return (
    <pre
      className="my-4 rounded-md border border-[var(--rule)] bg-white overflow-x-auto px-4 py-3 text-[13px] leading-relaxed"
      data-language={language}
    >
      <code className="font-mono text-[var(--foreground)]">{code}</code>
    </pre>
  );
}
