import Foundation

final class DataResourcesUtil {
    private init() {}
    
    static func loadGold(british: Bool) -> [String: Any] {
        let filename = british ? "gb_gold" : "us_gold"
        
      guard let url = Bundle.module.url(forResource: filename, withExtension: "json", subdirectory: "KittenTTS"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
            return [:]
        }
      
        return json
    }
    
    static func loadSilver(british: Bool) -> [String: Any] {
      let filename = british ? "gb_silver" : "us_silver"
      
      guard let url = Bundle.module.url(forResource: filename, withExtension: "json",  subdirectory: "KittenTTS"),
            let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
          return [:]
      }
            
      return json
    }
}
