import Foundation
import Security

enum HuggingFaceToken {
    private static var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: "\(Bundle.main.bundleIdentifier ?? "PhononTranslate").huggingface",
         kSecAttrAccount as String: "read-token"]
    }

    static func read() -> String? {
        var request = query
        request[kSecReturnData as String] = true
        request[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        if SecItemCopyMatching(request as CFDictionary, &item) == errSecSuccess,
           let data = item as? Data, let token = String(data: data, encoding: .utf8) {
            return token
        }
        return ProcessInfo.processInfo.environment["HF_TOKEN"]
    }

    static func save(_ token: String) throws {
        let token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if token.isEmpty {
            let status = SecItemDelete(query as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else { throw tokenError(status) }
            return
        }
        let data = Data(token.utf8)
        let status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item[kSecValueData as String] = data
            item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            let addStatus = SecItemAdd(item as CFDictionary, nil)
            guard addStatus == errSecSuccess else { throw tokenError(addStatus) }
        } else if status != errSecSuccess {
            throw tokenError(status)
        }
    }

    private static func tokenError(_ status: OSStatus) -> NSError {
        NSError(domain: NSOSStatusErrorDomain, code: Int(status), userInfo: [
            NSLocalizedDescriptionKey: "Could not save the Hugging Face token in Keychain."
        ])
    }
}
