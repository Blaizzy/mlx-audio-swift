import Foundation

final class DataResourcesUtil {
    private init() {}

    static func loadGold(british: Bool, directory: URL) -> [String: Any] {
        let filename = british ? "gb_gold" : "us_gold"
        return loadJSON(filename: filename, directory: directory, british: british)
    }

    static func loadSilver(british: Bool, directory: URL) -> [String: Any] {
        let filename = british ? "gb_silver" : "us_silver"
        return loadJSON(filename: filename, directory: directory, british: british)
    }

    private static func loadJSON(filename: String, directory: URL, british: Bool) -> [String: Any] {
        var url = directory.appendingPathComponent("\(filename).json")
        if !FileManager.default.fileExists(atPath: url.path), british {
            let fallback = filename.replacingOccurrences(of: "gb_", with: "us_")
            url = directory.appendingPathComponent("\(fallback).json")
        }
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
            return [:]
        }
        return json
    }
}
