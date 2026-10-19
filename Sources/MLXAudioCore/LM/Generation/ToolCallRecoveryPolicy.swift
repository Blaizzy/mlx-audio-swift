// Copyright © 2026 Apple Inc.

public enum ToolCallRecoveryPolicy: String, Hashable, Sendable, CaseIterable {
    /// Only the selected parser may produce calls. No alternate-dialect
    /// recovery is attempted.
    case disabled

    /// Recover only structurally complete alternate-dialect calls that carry
    /// an explicit protocol marker (`<tool_call>`, `<|tool_call>`,
    /// `<function=`, `[TOOL_CALLS]`) and appear in committed response text —
    /// never inside reasoning spans (`<think>`/`<thinking>`/`[THINK]`),
    /// Markdown code spans or fences, or ordinary JSON data — and only when
    /// the call names an exactly declared tool. Argument normalization and
    /// schema enforcement are governed independently by
    /// ``ToolCallValidationPolicy``.
    ///
    /// Explicit protocol attempts that are malformed, incomplete at end of
    /// stream, or undeclared are surfaced as ``RejectedToolCall`` rather than
    /// leaked as response text. Markerless `name[ARGS]{...}` rehearsals are
    /// never promoted by this policy: without a protocol marker, prose that
    /// merely mentions a declared call is indistinguishable from an intended
    /// one.
    case conservative

    /// Everything ``conservative`` does, plus promotion of exact markerless
    /// `declaredTool[ARGS]{...}` rehearsals in committed response text, and
    /// documented end-of-stream repair of calls whose outer closing marker is
    /// missing while their payload is structurally complete. Every promotion
    /// and repair is recorded in ``ToolCallProcessor/recoveryEvents`` with
    /// its provenance. Markerless candidates that fail validation remain
    /// response text.
    case permissive
}
