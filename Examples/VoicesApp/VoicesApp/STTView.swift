import SwiftUI
import UniformTypeIdentifiers
import Foundation

struct STTView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var viewModel = STTViewModel()
    @State private var showFileImporter = false
    @State private var showSettings = false
    @State private var isAudioHovering = false

    #if os(iOS)
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .callout
    private let bodyFont: Font = .body
    #else
    private let buttonHeight: CGFloat = 44
    private let buttonFont: Font = .subheadline
    private let bodyFont: Font = .title3
    #endif

    var body: some View {
        let transcriptSegments = TaggedTranscriptParser.parse(viewModel.transcriptionText)
        let activeTranscriptSegmentID = viewModel.isPlaying
            ? TaggedTranscriptParser.activeSegmentID(in: transcriptSegments, at: viewModel.currentTime)
            : nil

        VStack(spacing: 0) {
            // Transcription result area
            ScrollViewReader { proxy in
                ScrollView {
                    if viewModel.transcriptionText.isEmpty && !viewModel.isGenerating && !viewModel.isRecording {
                        VStack(spacing: 12) {
                            Spacer(minLength: 80)
                            Image(systemName: "waveform.badge.mic")
                                .font(.system(size: 48))
                                .foregroundStyle(.tertiary)
                            Text("Import or record audio to transcribe")
                                .font(bodyFont)
                                .foregroundStyle(.tertiary)
                            Spacer()
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if viewModel.transcriptionText.isEmpty && viewModel.isRecording {
                        RecordingTranscriptPlaceholder(supportsRealtime: viewModel.supportsRealtimeRecording)
                    } else {
                        TranscriptOutputView(
                            text: viewModel.transcriptionText,
                            font: bodyFont,
                            segments: transcriptSegments,
                            activeSegmentID: activeTranscriptSegmentID
                        )

                        Color.clear
                            .frame(height: 1)
                            .id("bottom")
                    }
                }
                .onChange(of: viewModel.transcriptionText) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
                .onChange(of: activeTranscriptSegmentID) { _, segmentID in
                    guard viewModel.isPlaying, let segmentID else { return }
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(segmentID, anchor: .center)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Recording indicator
            if viewModel.isRecording {
                RecordingIndicator(
                    duration: viewModel.recordingDuration,
                    audioLevel: viewModel.audioLevel,
                    supportsRealtime: viewModel.supportsRealtimeRecording,
                    isRealtime: viewModel.usesRealtimeRecording
                )
                .padding(.horizontal)
                .padding(.bottom, 4)
            }

            // Audio file info + player (hidden while recording)
            if viewModel.selectedAudioURL != nil && !viewModel.isRecording {
                VStack(spacing: 4) {
                    // File name
                    if let fileName = viewModel.audioFileName {
                        HStack(spacing: 6) {
                            Image(systemName: "doc.fill")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                            Text(fileName)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                            Spacer()
                            Button(action: { viewModel.removeAudioFile() }) {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundStyle(isAudioHovering ? Color.red : Color.secondary)
                                    .frame(width: 28, height: 28)
                                    .contentShape(Circle())
                            }
                            .buttonStyle(.plain)
                            .disabled(viewModel.isGenerating)
                            .accessibilityLabel("Remove audio")
                            .help("Remove audio")
                        }
                        .padding(.horizontal)
                        .padding(.top, 6)
                    }

                    // Audio player
                    CompactAudioPlayer(
                        isPlaying: viewModel.isPlaying,
                        currentTime: viewModel.currentTime,
                        duration: viewModel.duration,
                        onPlayPause: { viewModel.togglePlayPause() },
                        onSeek: { viewModel.seek(to: $0) }
                    )
                    .padding(.horizontal)
                }
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isAudioHovering ? Color.gray.opacity(0.12) : Color.clear)
                        .padding(.horizontal, 8)
                )
                .contentShape(Rectangle())
                .onHover { hovering in
                    isAudioHovering = hovering
                }
                .animation(.easeOut(duration: 0.12), value: isAudioHovering)
                .padding(.bottom, 4)
            }

            // Status/Progress
            VStack(spacing: 4) {
                if !viewModel.generationProgress.isEmpty && !viewModel.isRecording {
                    HStack(spacing: 6) {
                        ProgressView()
                            .scaleEffect(0.6)
                        Text(viewModel.generationProgress)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
                }

                if let error = viewModel.errorMessage {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom, 4)

            // Bottom bar
            HStack(spacing: 8) {
                if !viewModel.isRecording {
                    // File import button
                    Button(action: { showFileImporter = true }) {
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 6) {
                                Image(systemName: "doc.badge.plus")
                                Text("Import")
                            }
                            .font(buttonFont)
                            .foregroundStyle(.primary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 12)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())

                            Image(systemName: "doc.badge.plus")
                                .font(buttonFont)
                                .foregroundStyle(.primary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                .background(Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isGenerating)
                }

                // Record button
                Button(action: {
                    if viewModel.isRecording {
                        viewModel.stopRecording()
                    } else {
                        Task { await viewModel.startRecording() }
                    }
                }) {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 6) {
                            Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                            Text(viewModel.isRecording ? "Stop" : "Record")
                        }
                        .font(buttonFont)
                        .fontWeight(viewModel.isRecording ? .medium : .regular)
                        .foregroundStyle(viewModel.isRecording ? .white : .red)
                        .frame(height: buttonHeight)
                        .padding(.horizontal, 12)
                        .background(viewModel.isRecording ? Color.red : Color.gray.opacity(0.2))
                        .clipShape(Capsule())

                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                            .font(buttonFont)
                            .foregroundStyle(viewModel.isRecording ? .white : .primary)
                            .frame(width: buttonHeight, height: buttonHeight)
                            .background(viewModel.isRecording ? Color.red : Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.canRecord)

                if !viewModel.isRecording {
                    // Settings button
                    Button(action: { showSettings = true }) {
                        Image(systemName: "slider.horizontal.3")
                            .font(buttonFont)
                            .foregroundStyle(.primary)
                            .frame(width: buttonHeight, height: buttonHeight)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)

                    // Copy button (when transcription exists)
                    if !viewModel.transcriptionText.isEmpty {
                        Button(action: { viewModel.copyTranscription() }) {
                            Image(systemName: "doc.on.doc")
                                .font(buttonFont)
                                .foregroundStyle(.primary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                .background(Color.gray.opacity(0.2))
                                .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)

                        if !viewModel.isGenerating {
                            Button(action: { viewModel.clearTranscription() }) {
                                Image(systemName: "trash")
                                    .font(buttonFont)
                                    .foregroundStyle(.red)
                                    .frame(width: buttonHeight, height: buttonHeight)
                                    .background(Color.gray.opacity(0.2))
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Clear transcription")
                            .help("Clear transcription")
                        }
                    }

                    // Stats
                    if viewModel.tokensPerSecond > 0 {
                        HStack(spacing: 8) {
                            Label(
                                String(format: "%.1f tok/s", viewModel.tokensPerSecond),
                                systemImage: "speedometer"
                            )
                            Label(
                                String(format: "%.1f GB", viewModel.peakMemory),
                                systemImage: "memorychip"
                            )
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    }
                } else {
                    // Cancel button while recording
                    Button(action: { viewModel.cancelRecording() }) {
                        Text("Cancel")
                            .font(buttonFont)
                            .foregroundStyle(.secondary)
                            .frame(height: buttonHeight)
                            .padding(.horizontal, 12)
                            .background(Color.gray.opacity(0.2))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)

                    // Stats during recording
                    if viewModel.tokensPerSecond > 0 {
                        HStack(spacing: 8) {
                            Label(
                                String(format: "%.1f tok/s", viewModel.tokensPerSecond),
                                systemImage: "speedometer"
                            )
                            Label(
                                String(format: "%.1f GB", viewModel.peakMemory),
                                systemImage: "memorychip"
                            )
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                    }
                }

                Spacer()

                // Transcribe / Stop button (for file transcription, not shown during recording)
                if !viewModel.isRecording {
                    if viewModel.isGenerating {
                        Button(action: {
                            viewModel.stop()
                        }) {
                            ViewThatFits(in: .horizontal) {
                                Text("Stop")
                                    .font(buttonFont)
                                    .fontWeight(.medium)
                                    .foregroundStyle(.white)
                                    .frame(height: buttonHeight)
                                    .padding(.horizontal, 16)
                                    .background(Color.red)
                                    .clipShape(Capsule())

                                Image(systemName: "stop.fill")
                                    .font(buttonFont)
                                    .foregroundStyle(.white)
                                    .frame(width: buttonHeight, height: buttonHeight)
                                    .background(Color.red)
                                    .clipShape(Capsule())
                            }
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button(action: {
                            viewModel.startTranscription()
                        }) {
                            Image(systemName: "waveform.badge.mic")
                                .font(buttonFont)
                                .foregroundStyle(canTranscribe ? .white : .secondary)
                                .frame(width: buttonHeight, height: buttonHeight)
                                    .background(canTranscribe ? Color.blue : Color.gray.opacity(0.2))
                                    .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                        .disabled(!canTranscribe)
                    }
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            #if os(iOS)
            .background(Color(uiColor: .systemBackground).opacity(0.95))
            #else
            .background(.bar)
            #endif
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.audio, .wav, .mp3, .mpeg4Audio, .aiff],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    if url.startAccessingSecurityScopedResource() {
                        viewModel.selectAudioFile(url)
                    }
                }
            case .failure(let error):
                viewModel.errorMessage = "File import failed: \(error.localizedDescription)"
            }
        }
        .sheet(isPresented: $showSettings) {
            STTSettingsView(viewModel: viewModel)
                #if os(iOS)
                .presentationDetents([.large])
                .presentationDragIndicator(.visible)
                #endif
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background:
                viewModel.pause()
                viewModel.stop()
            default:
                break
            }
        }
        .task {
            await viewModel.loadModel()
        }
    }

    private var canTranscribe: Bool {
        viewModel.selectedAudioURL != nil && !viewModel.isGenerating && viewModel.isModelLoaded
    }
}

// MARK: - Transcript Output

private struct TranscriptOutputView: View {
    let text: String
    let font: Font
    let segments: [TaggedTranscriptSegment]
    let activeSegmentID: TaggedTranscriptSegment.ID?

    var body: some View {
        if segments.isEmpty {
            Text(text)
                .font(font)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        } else {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(segments) { segment in
                    TaggedTranscriptSegmentRow(
                        segment: segment,
                        font: font,
                        isActive: segment.id == activeSegmentID
                    )
                    .id(segment.id)
                }
            }
            .padding()
            .textSelection(.enabled)
        }
    }
}

private struct TaggedTranscriptSegmentRow: View {
    let segment: TaggedTranscriptSegment
    let font: Font
    let isActive: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Capsule()
                .fill(speakerColor)
                .frame(width: 4)
                .padding(.vertical, 2)
                .opacity(isActive ? 1 : 0)

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(segment.speaker)
                        .font(.caption.monospaced().weight(.semibold))
                        .foregroundStyle(speakerColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(speakerColor.opacity(isActive ? 0.22 : 0.14))
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

                    Text(segment.formattedTimeRange)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(isActive ? .primary : .secondary)

                    Spacer(minLength: 0)
                }

                Text(segment.text)
                    .font(font)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isActive ? speakerColor.opacity(0.12) : Color.primary.opacity(0.035))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(speakerColor.opacity(isActive ? 0.6 : 0.14), lineWidth: isActive ? 1.5 : 1)
        }
        .animation(.easeOut(duration: 0.16), value: isActive)
    }

    private var speakerColor: Color {
        let palette: [Color] = [
            .blue,
            .green,
            .orange,
            .purple,
            .pink,
            .teal,
        ]
        let number = Int(segment.speaker.drop { !$0.isNumber }) ?? 1
        return palette[(max(number, 1) - 1) % palette.count]
    }
}

private struct TaggedTranscriptSegment: Identifiable {
    let id: Int
    let speaker: String
    let startTime: String
    let startSeconds: TimeInterval
    let endTime: String?
    let endSeconds: TimeInterval?
    let text: String

    var formattedTimeRange: String {
        if let endTime {
            return "\(Self.formatTimestamp(startTime)) - \(Self.formatTimestamp(endTime))"
        }
        return Self.formatTimestamp(startTime)
    }

    private static func formatTimestamp(_ rawValue: String) -> String {
        guard let totalSeconds = timestampValue(rawValue) else { return rawValue }

        let minutes = Int(totalSeconds) / 60
        let seconds = totalSeconds - Double(minutes * 60)
        return String(
            format: "%d:%05.2f",
            locale: Locale(identifier: "en_US_POSIX"),
            minutes,
            seconds
        )
    }

    static func timestampValue(_ text: String) -> Double? {
        Double(text.replacingOccurrences(of: ",", with: "."))
    }
}

private enum TaggedTranscriptParser {
    private static let startTagRegex = try! NSRegularExpression(
        pattern: #"\[(\d+(?:[\.,]\d+)?)\]\[(S\d{1,3})\]"#,
        options: [.caseInsensitive]
    )
    private static let trailingTimestampRegex = try! NSRegularExpression(
        pattern: #"\[(\d+(?:[\.,]\d+)?)\]\s*$"#,
        options: []
    )

    static func parse(_ text: String) -> [TaggedTranscriptSegment] {
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = startTagRegex.matches(in: text, range: fullRange)
        guard !matches.isEmpty else { return [] }

        return matches.enumerated().compactMap { index, match in
            guard
                let matchRange = Range(match.range, in: text),
                let startRange = Range(match.range(at: 1), in: text),
                let speakerRange = Range(match.range(at: 2), in: text),
                let startSeconds = TaggedTranscriptSegment.timestampValue(String(text[startRange]))
            else {
                return nil
            }

            let bodyEnd: String.Index
            if index + 1 < matches.count,
               let nextRange = Range(matches[index + 1].range, in: text) {
                bodyEnd = nextRange.lowerBound
            } else {
                bodyEnd = text.endIndex
            }

            var segmentText = String(text[matchRange.upperBound..<bodyEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let endTime = stripTrailingTimestamp(from: &segmentText)
            let endSeconds = endTime.flatMap(TaggedTranscriptSegment.timestampValue)

            guard !segmentText.isEmpty else { return nil }

            return TaggedTranscriptSegment(
                id: index,
                speaker: String(text[speakerRange]).uppercased(),
                startTime: String(text[startRange]),
                startSeconds: startSeconds,
                endTime: endTime,
                endSeconds: endSeconds,
                text: segmentText
            )
        }
    }

    static func activeSegmentID(in segments: [TaggedTranscriptSegment], at time: TimeInterval) -> TaggedTranscriptSegment.ID? {
        for (index, segment) in segments.enumerated() {
            let nextStart = index + 1 < segments.count ? segments[index + 1].startSeconds : nil
            let end = segment.endSeconds ?? nextStart

            if let end {
                if time >= segment.startSeconds && time < max(end, segment.startSeconds) {
                    return segment.id
                }
            } else if time >= segment.startSeconds {
                return segment.id
            }
        }

        return nil
    }

    private static func stripTrailingTimestamp(from text: inout String) -> String? {
        var lastTimestamp: String?

        while true {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard
                let match = trailingTimestampRegex.firstMatch(in: text, range: range),
                let timestampRange = Range(match.range(at: 1), in: text),
                let fullMatchRange = Range(match.range, in: text)
            else {
                return lastTimestamp
            }

            lastTimestamp = String(text[timestampRange])
            text.removeSubrange(fullMatchRange)
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
}

// MARK: - Recording Indicator

private struct RecordingTranscriptPlaceholder: View {
    let supportsRealtime: Bool

    var body: some View {
        VStack(spacing: 8) {
            Spacer(minLength: 80)

            Image(systemName: supportsRealtime ? "text.bubble.fill" : "waveform.circle.fill")
                .font(.system(size: 40, weight: .medium))
                .foregroundStyle(.tertiary)

            Text(supportsRealtime ? "Listening..." : "Recording...")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(supportsRealtime ? "Realtime prediction will appear here" : "Starting realtime preview")
                .font(.caption)
                .foregroundStyle(.tertiary)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

private struct RecordingIndicator: View {
    let duration: TimeInterval
    let audioLevel: Float
    let supportsRealtime: Bool
    let isRealtime: Bool

    private let barCount = 28

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.red.opacity(0.14))
                    .frame(width: 34, height: 34)

                Image(systemName: "mic.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.red)
                    .symbolEffect(.pulse, options: .repeating, value: duration)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(isRealtime ? "Live" : "Recording")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Text(supportsRealtime ? "Realtime preview" : "Starting...")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(minWidth: 88, alignment: .leading)

            TimelineView(.animation) { timeline in
                let phase = timeline.date.timeIntervalSinceReferenceDate * 5.8
                let level = max(0, min(CGFloat(audioLevel), 1))

                HStack(alignment: .center, spacing: 3) {
                    ForEach(0..<barCount, id: \.self) { index in
                        let position = CGFloat(index) / CGFloat(max(barCount - 1, 1))
                        let ripple = abs(sin(phase + Double(index) * 0.47))
                        let envelope = 0.35 + 0.65 * sin(position * .pi)
                        let idleLift = 0.10 + 0.10 * CGFloat(ripple)
                        let reactiveLift = level * CGFloat(0.35 + 0.65 * ripple) * envelope
                        let height = 8 + (idleLift + reactiveLift) * 34

                        Capsule()
                            .fill(barGradient(at: position))
                            .frame(width: 4, height: height)
                            .shadow(color: Color.red.opacity(0.22 * (idleLift + reactiveLift)), radius: 6)
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .animation(.easeOut(duration: 0.12), value: audioLevel)
            }

            Spacer()

            Text(formatDuration(duration))
                .font(.caption.monospacedDigit().weight(.medium))
                .foregroundStyle(.secondary)
                .frame(minWidth: 42, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.red.opacity(0.07))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.red.opacity(0.16), lineWidth: 1)
        }
    }

    private func barGradient(at position: CGFloat) -> LinearGradient {
        let leading = Color(red: 1.0, green: 0.23, blue: 0.30)
        let center = Color(red: 1.0, green: 0.42, blue: 0.58)
        let trailing = Color(red: 0.72, green: 0.33, blue: 1.0)
        let color = position < 0.5
            ? leading.mix(with: center, by: position * 2)
            : center.mix(with: trailing, by: (position - 0.5) * 2)

        return LinearGradient(
            colors: [color.opacity(0.65), color, color.opacity(0.75)],
            startPoint: .bottom,
            endPoint: .top
        )
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

private extension Color {
    func mix(with other: Color, by amount: CGFloat) -> Color {
        #if os(iOS)
        let first = UIColor(self)
        let second = UIColor(other)
        #else
        let first = NSColor(self)
        let second = NSColor(other)
        #endif

        var red1: CGFloat = 0
        var green1: CGFloat = 0
        var blue1: CGFloat = 0
        var alpha1: CGFloat = 0
        var red2: CGFloat = 0
        var green2: CGFloat = 0
        var blue2: CGFloat = 0
        var alpha2: CGFloat = 0

        first.getRed(&red1, green: &green1, blue: &blue1, alpha: &alpha1)
        second.getRed(&red2, green: &green2, blue: &blue2, alpha: &alpha2)

        let clampedAmount = max(0, min(amount, 1))
        return Color(
            red: red1 + (red2 - red1) * clampedAmount,
            green: green1 + (green2 - green1) * clampedAmount,
            blue: blue1 + (blue2 - blue1) * clampedAmount,
            opacity: alpha1 + (alpha2 - alpha1) * clampedAmount
        )
    }
}

#Preview {
    STTView()
}
