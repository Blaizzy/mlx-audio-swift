# Examples

## SimpleChat Initial Setup

Create a local config file:

```sh
cp Examples/SimpleChat/Config/Local.xcconfig.template Examples/SimpleChat/Config/Local.xcconfig
```

Edit `Examples/SimpleChat/Config/Local.xcconfig` and set:

`APP_BUNDLE_ID` to something unique (for example, `com.yourname.SimpleChat`)

`DEVELOPMENT_TEAM` to your Apple Developer Team ID

Open `Examples/SimpleChat/SimpleChat.xcodeproj` in Xcode and build `SimpleChat`.

## Simulator / CI (no signing)

Use this for Simulator builds in CI without signing:

```sh
xcodebuild \
  -project Examples/SimpleChat/SimpleChat.xcodeproj \
  -scheme SimpleChat \
  -sdk iphonesimulator \
  -configuration Debug \
  CODE_SIGNING_ALLOWED=NO
```

## Notes

- The app asks for microphone permission on first use.
- On first run, on-device speech assets may need to download before speech detection is ready.
