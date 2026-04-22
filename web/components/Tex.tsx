import katex from "katex";

type Props = {
  expr: string;
  display?: boolean;
};

export default function Tex({ expr, display = false }: Props) {
  const html = katex.renderToString(expr, {
    displayMode: display,
    throwOnError: false,
    strict: "ignore",
  });

  if (display) {
    return (
      <div
        className="overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
