// Back-compat aliases for the shared NeMo-family types extracted into
// `Models/Nemo/` (issue #197). These keep `ParakeetModel`, `ParakeetConformer`,
// the rest of `ParakeetConfig`, and any external code compiling unchanged.
// Access levels mirror the underlying Nemo types.

import Foundation

// Attention (NemoAttention.swift) — internal
typealias ParakeetMultiHeadAttention = NemoMultiHeadAttention
typealias ParakeetRelPositionMultiHeadAttention = NemoRelPositionMultiHeadAttention
typealias ParakeetRelPositionalEncoding = NemoRelPositionalEncoding

// RNN-T layers (NemoRNNTLayers.swift) — internal
typealias ParakeetLSTMState = NemoLSTMState
typealias ParakeetPredictNetwork = NemoPredictNetwork
typealias ParakeetJointNetwork = NemoJointNetwork

// Decoding (NemoDecodingLogic.swift) — internal
typealias ParakeetDecodingLogic = NemoDecodingLogic

// Alignment (NemoAlignment.swift)
public typealias ParakeetAlignedToken = NemoAlignedToken
public typealias ParakeetAlignedSentence = NemoAlignedSentence
public typealias ParakeetAlignedResult = NemoAlignedResult
public typealias ParakeetStreamingResult = NemoStreamingResult
typealias ParakeetAlignment = NemoAlignment
typealias ParakeetAlignmentError = NemoAlignmentError

// RNN-T config structs (NemoRNNTConfig.swift) — public
public typealias ParakeetPredictNetworkConfig = NemoPredictNetworkConfig
public typealias ParakeetPredictConfig = NemoPredictConfig
public typealias ParakeetJointNetworkConfig = NemoJointNetworkConfig
public typealias ParakeetJointConfig = NemoJointConfig
