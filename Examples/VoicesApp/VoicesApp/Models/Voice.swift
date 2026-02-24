import Foundation
import SwiftUI

struct Voice: Identifiable, Hashable {
    let id: UUID
    var name: String
    var description: String
    var language: String
    var color: Color
    var isCustom: Bool
    var lastUsed: Date?

    // Voice cloning properties
    var audioFileURL: URL?
    var transcription: String?

    /// Whether this voice uses audio cloning (has both audio file and transcription)
    var isClonedVoice: Bool {
        audioFileURL != nil && transcription != nil && !transcription!.isEmpty
    }

    init(
        id: UUID = UUID(),
        name: String,
        description: String = "",
        language: String = "English",
        color: Color = .blue,
        isCustom: Bool = false,
        lastUsed: Date? = nil,
        audioFileURL: URL? = nil,
        transcription: String? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.language = language
        self.color = color
        self.isCustom = isCustom
        self.lastUsed = lastUsed
        self.audioFileURL = audioFileURL
        self.transcription = transcription
    }
}

struct VoiceCollection: Identifiable {
    let id: UUID
    var name: String
    var description: String
    var imageName: String
    var backgroundColor: Color
    var voices: [Voice]

    init(
        id: UUID = UUID(),
        name: String,
        description: String = "",
        imageName: String = "",
        backgroundColor: Color = .pink.opacity(0.3),
        voices: [Voice] = []
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.imageName = imageName
        self.backgroundColor = backgroundColor
        self.voices = voices
    }
}

// MARK: - Sample Data

extension Voice {
    static let samples: [Voice] = [
        Voice(
            name: "Lily",
            description: "Velvety Actress",
            language: "English",
            color: .purple.opacity(0.3),
            lastUsed: Date(),
            audioFileURL: Bundle.main.url(forResource: "Lily", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        ),
        Voice(
            name: "James",
            description: "Professional Narrator",
            language: "English",
            color: .blue.opacity(0.3),
            lastUsed: Date().addingTimeInterval(-3600),
            audioFileURL: Bundle.main.url(forResource: "James", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        ),
        Voice(
            name: "Sophie",
            description: "Warm & Friendly",
            language: "English",
            color: .orange.opacity(0.3),
            lastUsed: Date().addingTimeInterval(-7200),
            audioFileURL: Bundle.main.url(forResource: "Sophie", withExtension: "wav"),
            transcription: "Judge Victor quickly mixed a beige waxy potion for the zesty, grumpy elf who yawns."
        )
    ]

    static let customVoices: [Voice] = [
        Voice(
            name: "German action voice",
            description: "Custom voice",
            language: "English",
            color: .teal.opacity(0.3),
            isCustom: true
        ),
        Voice(
            name: "Bane",
            description: "Deep & Dramatic",
            language: "English",
            color: .green.opacity(0.3),
            isCustom: true
        )
    ]
}

extension VoiceCollection {
    static let samples: [VoiceCollection] = [
        VoiceCollection(
            name: "Father Christmas & Characters",
            description: "Holiday themed voices",
            imageName: "gift.fill",
            backgroundColor: .pink.opacity(0.2)
        ),
        VoiceCollection(
            name: "Professional Narrators",
            description: "For audiobooks and podcasts",
            imageName: "book.fill",
            backgroundColor: .purple.opacity(0.2)
        ),
        VoiceCollection(
            name: "Character Voices",
            description: "Unique personalities",
            imageName: "theatermasks.fill",
            backgroundColor: .blue.opacity(0.2)
        )
    ]
}
