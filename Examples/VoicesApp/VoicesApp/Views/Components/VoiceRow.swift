import SwiftUI

struct VoiceRow: View {
    let voice: Voice
    var showDeleteButton: Bool = false
    var onDelete: (() -> Void)?
    var onTap: (() -> Void)?

    var body: some View {
        Button(action: { onTap?() }) {
            HStack(spacing: 12) {
                // Voice avatar
                VoiceAvatar(color: voice.color, size: 44)

                VStack(alignment: .leading, spacing: 2) {
                    Text(voice.name)
                        .font(.body)
                        .fontWeight(.medium)
                        .foregroundStyle(.primary)

                    if !voice.description.isEmpty {
                        Text(voice.description)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else {
                        Text(voice.language)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                if showDeleteButton {
                    Button(action: { onDelete?() }) {
                        Image(systemName: "minus.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct VoiceAvatar: View {
    let color: Color
    var size: CGFloat = 44

    var body: some View {
        ZStack {
            Circle()
                .fill(color.gradient)
                .frame(width: size, height: size)

            // Swirl pattern
            Circle()
                .fill(
                    AngularGradient(
                        colors: [
                            color.opacity(0.8),
                            color.opacity(0.4),
                            color.opacity(0.8)
                        ],
                        center: .center,
                        startAngle: .degrees(0),
                        endAngle: .degrees(360)
                    )
                )
                .frame(width: size * 0.8, height: size * 0.8)
        }
    }
}

struct VoiceChip: View {
    let voice: Voice
    var onTap: (() -> Void)?

    var body: some View {
        Button(action: { onTap?() }) {
            HStack(spacing: 8) {
                VoiceAvatar(color: voice.color, size: 28)

                Text("\(voice.name) - \(voice.description)")
                    .font(.subheadline)
                    .foregroundStyle(.primary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.gray.opacity(0.15))
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    VStack(spacing: 20) {
        VoiceRow(voice: Voice.samples[0])

        VoiceRow(
            voice: Voice.customVoices[0],
            showDeleteButton: true,
            onDelete: {}
        )

        VoiceChip(voice: Voice.samples[0])
    }
    .padding()
}
