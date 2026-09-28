/** A stack of indexed planes: the product mark shares the diagrams' geometry. */
export function TensorMark() {
  return (
    <svg
      className="tensor-mark"
      viewBox="0 0 40 40"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M13 5h20v20M9 10h20v20"
        stroke="currentColor"
        strokeOpacity=".45"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <rect
        x="5"
        y="15"
        width="20"
        height="20"
        rx="4"
        fill="currentColor"
        fillOpacity=".08"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="M15 15v20M5 25h20" stroke="currentColor" strokeOpacity=".6" />
      <rect
        x="16.5"
        y="26.5"
        width="6.5"
        height="6.5"
        rx="1.5"
        fill="#b5b7eb"
      />
    </svg>
  );
}
