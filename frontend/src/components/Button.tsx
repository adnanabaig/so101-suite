import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";

type Variant = "primary" | "secondary" | "danger" | "ghost" | "success";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  children: ReactNode;
}

const variants: Record<Variant, string> = {
  primary:
    "bg-cyan-500 hover:bg-cyan-400 text-slate-950 border border-cyan-400/40",
  secondary:
    "bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700",
  // White on red-600 is 4.8:1; hover darkens (red-700, 6.5:1) instead of lightening to 3.8:1.
  danger: "bg-red-600 hover:bg-red-700 text-white border border-red-500/50",
  success:
    "bg-emerald-500 hover:bg-emerald-400 text-slate-950 border border-emerald-400/40",
  ghost:
    "bg-transparent hover:bg-slate-800 text-slate-300 border border-transparent",
};

const sizes: Record<Size, string> = {
  sm: "px-2.5 py-1 text-xs",
  md: "px-3.5 py-1.5 text-sm",
  lg: "px-5 py-2.5 text-base",
};

const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", className = "", children, disabled, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={`inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:opacity-40 ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...rest}
    >
      {children}
    </button>
  );
});

export default Button;
