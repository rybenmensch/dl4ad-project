from lib import check_path
from rave_lib import RAVEModel, rave_from_checkpoint

model = rave_from_checkpoint("models/satyr/")
source_path = check_path("./audio/source/")
reconstructed_path = check_path("./audio/")

file_name: str = "GLM.wav"

# input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
# base_source, sr = torchaudio.load(input_path)
# base_reconstruction = process_audio(model_clean, base_source)

model = RAVEModel(model)

print(model.model)
layer = model.from_net_path.get_layer("encoder.encoder.net", 1)

# layer = from_path.get_layer(model, "encoder.encoder.net.0")

# from_path.get_layer(model, "decoder.net.0")


# encoder_module_paths = collect_weight_module_paths(
#     model_clean,
#     "encoder",
# )
#
# depth_positions = {
#     "early": 0.10,
#     "middle": 0.50,
#     "late": 0.90,
# }
#
# encoder_targets = select_representative_depths(
#     encoder_module_paths,
#     depth_positions,
# )
#
# zc = calculate_zero_crossing_rate(base_source)
#
# a = np.array([0, 1, 2, 3])
# signs = np.sign(a)
#
# print(signs)
#
# print(a[1:])
# print(a[:-1])
# crossings = signs[1:] != signs[:-1]
# print(crossings)
