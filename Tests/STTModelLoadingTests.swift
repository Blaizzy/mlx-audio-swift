import Foundation
import Testing
@testable import MLXAudioSTT

struct STTModelLoadingTests {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "whisper-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test(arguments: ["model_type", "architecture", "model_version"])
    func localConfigDeterminesModelType(key: String) async throws {
        let url = try directory()
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: [key: "  FUTURE_ASR  "])
            .write(to: url.appendingPathComponent("config.json"))

        for explicitNil in [false, true] {
            do {
                if explicitNil {
                    _ = try await STT.loadModel(modelRepo: url.path, modelType: nil)
                } else {
                    _ = try await STT.loadModel(modelRepo: url.path)
                }
                Issue.record("Expected unsupported local model type")
            } catch STTModelError.unsupportedModelType(let type) {
                #expect(type == "future_asr")
            }
        }
    }

    @Test func localConfigUsesModelTypeBeforeArchitecture() async throws {
        let url = try directory()
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: [
            "model_type": "primary_asr", "architecture": "secondary_asr",
        ]).write(to: url.appendingPathComponent("config.json"))

        do {
            _ = try await STT.loadModel(modelRepo: url.path)
            Issue.record("Expected unsupported local model type")
        } catch STTModelError.unsupportedModelType(let type) {
            #expect(type == "primary_asr")
        }
    }

    @Test func localConfigErrorsAreNotTreatedAsHubRepositories() async throws {
        let url = try directory()
        defer { try? FileManager.default.removeItem(at: url) }
        for malformed in [false, true] {
            if malformed {
                try Data("{".utf8).write(to: url.appendingPathComponent("config.json"))
            }
            do {
                _ = try await STT.loadModel(modelRepo: url.path)
                Issue.record("Expected a local config read or parse error")
            } catch {
                #expect((error as NSError).domain == NSCocoaErrorDomain)
                if !malformed {
                    #expect((error as NSError).code == CocoaError.fileReadNoSuchFile.rawValue)
                }
            }
        }
    }
}
