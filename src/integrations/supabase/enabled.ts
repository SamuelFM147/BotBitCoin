// Centraliza a checagem de habilitação do Supabase no frontend
export function isSupabaseEnabled(): boolean {
  const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
  const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string | undefined;
  const disabled = (import.meta.env.VITE_DISABLE_SUPABASE as string | undefined)?.toLowerCase();

  if (disabled === "true" || disabled === "1") return false;
  if (!url || !key) return false;
  return true;
}