/**
 * labels.js -- Pure helpers for answer-mode labels and source badges.
 * No DOM access. Importable by both app.js and Node tests.
 */

/**
 * Return the user-facing label for the answer mode, or null if none needed.
 *
 * @param {string} route - From the start event
 * @param {string} answerMode - From the end event
 * @returns {{ text: string, style: string } | null}
 *   style: "muted" | "neutral" | "info" | "warn"
 */
export function getAnswerModeLabel(route, answerMode) {
  if (answerMode === "refusal") {
    if (route === "out_of_scope") {
      return { text: "Outside what I can help with", style: "muted" };
    }
    return { text: "No matching listings", style: "neutral" };
  }
  if (answerMode === "retrieval_only") {
    return {
      text: "Model busy: showing matching listings without a written summary",
      style: "info",
    };
  }
  if (answerMode === "static_busy") {
    return { text: "High traffic, please retry shortly", style: "warn" };
  }
  // llm and cache: no visible label (cache shows tag inside details only)
  return null;
}

/**
 * Return a display name for the data source.
 * @param {string} source - "wasalt" or "darglobal"
 * @returns {string}
 */
export function sourceBadgeLabel(source) {
  if (source === "wasalt") return "Wasalt";
  if (source === "darglobal") return "Dar Global";
  return source || "";
}
