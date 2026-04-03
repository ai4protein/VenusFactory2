type SegmentedOption<T extends string> = {
  value: T;
  label: string;
};

type SegmentedSwitchProps<T extends string> = {
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  ariaLabel: string;
  className?: string;
};

export function SegmentedSwitch<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  className = ""
}: SegmentedSwitchProps<T>) {
  return (
    <div className={`custom-segment-switch ${className}`.trim()} role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={`custom-segment-option ${value === option.value ? "active" : ""}`}
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
