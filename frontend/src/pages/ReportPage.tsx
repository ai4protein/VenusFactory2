import { Navigate } from "react-router-dom";
import { useLang } from "../lib/i18n";
export function ReportPage(_props: { workspaceEnabled?: boolean }) { const { lang } = useLang(); return <Navigate replace to={`/${lang}/agent/chat?mode=researcher`}/>; }
