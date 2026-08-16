# Examples

## SimpleChat Initial Setup

Generate the Xcode project from the XcodeGen spec:

```sh
cd Examples/SimpleChat
xcodegen generate
```

Create a local config file:

```sh
cp Examples/SimpleChat/Config/Local.xcconfig.template Examples/SimpleChat/Config/Local.xcconfig
```

Edit `Examples/SimpleChat/Config/Local.xcconfig` and set:

`APP_BUNDLE_ID` to something unique (for example, `com.yourname.SimpleChat`)

`DEVELOPMENT_TEAM` to your Apple Developer Team ID

Open `Examples/SimpleChat/SimpleChat.xcodeproj` in Xcode and build `SimpleChat`.

The Swift package entrypoint is still available:

```sh
cd Examples/SimpleChat
swift run SimpleChat
```

## Simulator / CI (no signing)

Use this for Simulator builds in CI without signing:

```sh
cd Examples/SimpleChat
xcodegen generate
xcodebuild \
  -project SimpleChat.xcodeproj \
  -scheme SimpleChat \
  -destination 'generic/platform=iOS Simulator' \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO
```

Use this for a macOS Xcode build without signing:

```sh
cd Examples/SimpleChat
xcodegen generate
xcodebuild \
  -project SimpleChat.xcodeproj \
  -scheme SimpleChat \
  -destination 'platform=macOS' \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO
```

## Notes

- The app asks for microphone permission on first use.
- On first run, on-device speech assets may need to download before speech detection is ready.
