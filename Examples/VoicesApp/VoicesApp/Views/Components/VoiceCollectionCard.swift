import SwiftUI

struct VoiceCollectionCard: View {
    let collection: VoiceCollection
    var onTap: (() -> Void)?

    var body: some View {
        Button(action: { onTap?() }) {
            HStack(spacing: 0) {
                // Image placeholder
                ZStack {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))

                    Image(systemName: collection.imageName)
                        .font(.system(size: 40))
                        .foregroundStyle(.secondary)
                }
                .frame(width: 140, height: 160)
                .clipShape(RoundedRectangle(cornerRadius: 12))

                // Title
                VStack(alignment: .leading) {
                    Text(collection.name)
                        .font(.headline)
                        .foregroundStyle(.primary)
                        .multilineTextAlignment(.leading)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 160)
            .background(collection.backgroundColor)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
    }
}

struct CollectionsSection: View {
    let collections: [VoiceCollection]
    var onCollectionTap: ((VoiceCollection) -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Voice collections")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("Best of the best, for your projects")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(collections) { collection in
                        VoiceCollectionCard(collection: collection) {
                            onCollectionTap?(collection)
                        }
                        .frame(width: 300)
                    }
                }
            }
        }
    }
}

#Preview {
    CollectionsSection(collections: VoiceCollection.samples)
        .padding()
}
