# Swift Logger Log Levels

> Research Date: 2026-01-08

## Two Main Logging Systems in Swift

| System | Package | Use Case |
|--------|---------|----------|
| **OSLog / os.Logger** | Built-in (Apple) | iOS/macOS apps, Console.app integration |
| **SwiftLog** | `apple/swift-log` | Server-side Swift, cross-platform, libraries |

---

## OSLog Log Levels (Apple's Unified Logging)

| Level | Method | Persistence | Use Case |
|-------|--------|-------------|----------|
| **debug** | `.debug()` | Only with debugger attached | Development-only verbose info |
| **info** | `.info()` | Only with debugger attached | Helpful but non-essential info |
| **notice** | `.notice()` / `.log()` | Persisted | Default level, notable events |
| **error** | `.error()` | Persisted | Error conditions |
| **fault** | `.fault()` | Persisted | System-level/multi-process errors |
| **critical** | `.critical()` | Persisted | Critical failures |

### Usage Example

```swift
import OSLog

extension Logger {
    private static var subsystem = Bundle.main.bundleIdentifier!
    static let network = Logger(subsystem: subsystem, category: "network")
    static let audio = Logger(subsystem: subsystem, category: "audio")
}

// Usage
Logger.network.debug("Request started: \(url)")
Logger.network.info("Response received")
Logger.network.error("Connection failed: \(error)")
```

### Privacy in OSLog

```swift
// Public data (visible in Console.app)
logger.log("Status: \(newStatus, privacy: .public)")

// Private data with hash for correlation
logger.log("Token: \(accessToken, privacy: .private(mask: .hash))")
```

---

## SwiftLog Log Levels (Server-Side / Cross-Platform)

| Level | Severity | Description |
|-------|----------|-------------|
| **trace** | Lowest | Finest diagnostics, "log everything" |
| **debug** | Low | High-value operational info, production-safe |
| **info** | Medium | General informational messages |
| **notice** | Medium | Notable events |
| **warning** | High | Cautionary situations |
| **error** | Higher | Error conditions |
| **critical** | Highest | Severe failures, library stops functioning |

### Usage Example

```swift
import Logging

var logger = Logger(label: "com.myapp.audio")
logger.logLevel = .debug  // Set minimum level

logger.trace("Detailed state: \(state)")
logger.debug("Processing audio chunk")
logger.info("Session started")
logger.warning("Buffer underrun detected")
logger.error("Failed to load model: \(error)")
logger.critical("Fatal: Cannot recover")
```

### Structured Logging with Metadata

```swift
logger.debug("Database connection established", metadata: [
    "host": "\(host)",
    "database": "\(database)",
    "connectionTime": "\(duration)"
])

logger.trace("Connection pool state", metadata: [
    "active": "\(activeConnections)",
    "idle": "\(idleConnections)",
    "pending": "\(pendingRequests)"
])
```

---

## Changing Log Level at Runtime

### SwiftLog

```swift
var logger = Logger(label: "MyLogger")
logger.logLevel = .debug  // Only affects this instance
```

### OSLog

Log level is controlled by system (Console.app) or Xcode, not programmatically.

---

## Best Practices for Libraries

| Do | Don't |
|----|-------|
| Use `trace`/`debug` for most logging | Log normal operations at `info`+ |
| Let users decide error handling | Log expected failures as errors |
| Include metadata (correlation IDs) | Use multiline log messages |
| Keep `debug` production-safe | Mutate log handlers in library code |

### Examples

```swift
// ❌ Bad: floods production logs
logger.info("HTTP request received")
logger.info("Database query executed")
logger.info("Response sent")

// ✅ Good: appropriate levels
logger.debug("Processing request", metadata: ["path": "\(path)"])
logger.trace("Query: \(query)")
logger.debug("Request completed", metadata: ["status": "\(status)"])
```

---

## Choosing Between OSLog and SwiftLog

| Criteria | OSLog | SwiftLog |
|----------|-------|----------|
| Platform | Apple only | Cross-platform |
| Console.app integration | ✅ Native | Via adapter |
| Privacy controls | ✅ Built-in | Manual |
| Server-side Swift | ❌ | ✅ Recommended |
| Library development | Good | Better (ecosystem standard) |

For **mlx-audio-swift** (Apple-only SDK), OSLog is the natural choice with its native Console.app integration and privacy controls.

---

## References

- [Swift.org — Log Levels Guide](https://www.swift.org/documentation/server/guides/libraries/log-levels.html)
- [apple/swift-log — GitHub](https://github.com/apple/swift-log)
- [Donny Wals — Modern logging with OSLog](https://www.donnywals.com/modern-logging-with-the-oslog-framework-in-swift/)
- [SwiftLee — OSLog and Unified Logging](https://www.avanderlee.com/debugging/oslog-unified-logging/)
