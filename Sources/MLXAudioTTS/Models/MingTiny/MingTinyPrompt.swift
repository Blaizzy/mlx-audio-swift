import Foundation

public enum MingTinyTask: String, Sendable {
    case ip, seed, multilingual, paralinguistic, podcast, basic, emotion, dialect
    case voiceDesign = "voice_design", freeform
    var usesContinuation: Bool { [.seed, .multilingual, .paralinguistic, .podcast].contains(self) }
    var requiresSpeaker: Bool { [.basic, .emotion, .dialect].contains(self) }
}

public struct MingTinyOptions: Sendable {
    public var task: MingTinyTask?
    public var instruction: String?
    public var style: String?
    public var emotion: String?
    public var dialect: String?
    public var speed: String?
    public var pitch: String?
    public var volume: String?
    public var cfgScale: Float = 2
    public var diffusionSteps = 10
    public var sigma: Float = 0.25
    public var minimumDecodeSteps: Int?
    public var stopThreshold: Float?
    public init() {}
}

enum MingTinyPrompt {
    static let languages = ["zh": "Chinese", "chinese": "Chinese", "en": "English", "english": "English",
        "de": "German", "german": "German", "es": "Spanish", "spanish": "Spanish", "fr": "French",
        "french": "French", "ko": "Korean", "korean": "Korean", "ja": "Japanese", "japanese": "Japanese"]

    static func instruction(task: MingTinyTask, voice: String, options o: MingTinyOptions) throws -> String? {
        var controls = ["style": o.style, "emotion": o.emotion, "dialect": o.dialect,
            "speed": o.speed, "pitch": o.pitch, "volume": o.volume].compactMapValues { $0 }
        if task == .ip { controls["ip"] = voice }
        if task.usesContinuation {
            guard controls.isEmpty, o.instruction == nil else {
                throw MingTinyError.invalidInput("Continuation tasks do not accept style controls")
            }
            return nil
        }
        if task == .freeform {
            guard controls.isEmpty, let text = o.instruction, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw MingTinyError.invalidInput("Freeform requires an instruction and no structured controls")
            }
            return text.hasPrefix("Speaking instruction:") ? text : "Speaking instruction: Speak in this style: " + text
        }
        let required: Set<String> = task == .basic ? ["speed", "pitch", "volume"]
            : [task == .voiceDesign ? "style" : task.rawValue]
        guard o.instruction == nil, Set(controls.keys) == required,
              controls.values.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw MingTinyError.invalidInput("\(task.rawValue) requires exactly these controls: \(required.sorted())")
        }
        let dialects = ["cantonese": "广粤话", "sichuanese": "川渝话", "chuanyu dialect": "川渝话"]
        let emotions = ["happy": "高兴", "sad": "悲伤", "angry": "愤怒", "neutral": "中性",
                        "surprised": "惊讶", "fearful": "恐惧", "disgusted": "厌恶"]
        if let dialect = controls["dialect"] { controls["dialect"] = dialects[dialect.lowercased()] ?? dialect }
        if let emotion = controls["emotion"] { controls["emotion"] = emotions[emotion.lowercased()] ?? emotion }
        let allowed: [String: Set<String>] = ["dialect": ["广粤话", "川渝话"],
            "emotion": Set(emotions.values), "speed": ["慢速", "中速", "快速"], "pitch": ["低", "中", "高"], "volume": ["低", "中", "高"]]
        for (name, value) in controls where allowed[name].map({ !$0.contains(value) }) ?? false {
            throw MingTinyError.invalidInput("Unsupported \(name): \(value)")
        }
        func quote(_ value: String?) -> String {
            value.map { String(data: try! JSONEncoder().encode($0), encoding: .utf8)! } ?? "null"
        }
        let fields = task == .emotion ? [("方言", "dialect"), ("风格", "style"), ("语速", "speed"), ("基频", "pitch"), ("音量", "volume"), ("情感", "emotion")]
            : [("方言", "dialect"), ("情感", "emotion"), ("风格", "style"), ("语速", "speed"), ("基频", "pitch"), ("音量", "volume")]
        var values = ["\"序号\": 1", "\"说话人\": \"speaker_1\""]
        values += fields.map { quote($0.0) + ": " + quote(controls[$0.1]) }
        values.append("\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, \"ENV\": null, \"SNR\": null}")
        values.append("\"IP\": " + quote(controls["ip"]))
        return "{\"audio_sequence\": [{" + values.joined(separator: ", ") + "}]}"
    }

    static func build(encode: (String) -> [Int], speech s: MingTinySpeechConfiguration,
                      text: String, instruction: String?, language: String?, speakers: Int,
                      promptText: String?, promptSteps: Int) -> (ids: [Int], positions: [Int]) {
        var prefix = "<role>HUMAN</role>Please generate speech based on the following description.\n"
        if let language { prefix += "Target language: \(language).\n" }
        let assistantInstruction = instruction.map { !$0.hasPrefix("Speaking instruction:") } ?? false
        if let instruction, !assistantInstruction { prefix += instruction + "\n" }
        if speakers == 1 { prefix += "  speaker_1:" }
        var ids = encode(prefix), positions: [Int] = []
        for index in 0..<speakers {
            if speakers > 1 { ids += encode((index > 0 ? "\n" : "") + "  speaker_\(index + 1):") }
            ids.append(s.speakerStartTokenId); positions.append(ids.count)
            ids += [s.audioPatchTokenId, s.speakerEndTokenId]
        }
        ids += encode("\n Text input:\n")
        if let promptText { ids += encode(promptText) }
        ids += encode(text) + encode("<role>ASSISTANT</role>")
        if let instruction, assistantInstruction { ids += encode(instruction) + [s.promptEosTokenId] }
        ids += [s.audioStartTokenId] + Array(repeating: s.audioPatchTokenId, count: promptSteps)
        return (ids, positions)
    }

    static func validatePodcast(_ text: String, count: Int) throws {
        guard count >= 2 else { throw MingTinyError.invalidInput("Podcast requires at least two speakers") }
        let regex = try NSRegularExpression(pattern: "^\\s*speaker_(\\d+)\\s*:\\s*\\S")
        var seen = Set<Int>()
        for line in text.components(separatedBy: .newlines) where !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
                  let range = Range(match.range(at: 1), in: line), let index = Int(line[range]), (1...count).contains(index) else {
                throw MingTinyError.invalidInput("Every podcast line must start with a valid speaker_N:")
            }
            seen.insert(index)
        }
        guard seen == Set(1...count) else { throw MingTinyError.invalidInput("Podcast dialogue must include every reference speaker") }
    }
}
