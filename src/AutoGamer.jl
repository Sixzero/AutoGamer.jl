module AutoGamer

greet() = print("Hello World!")

include("data.jl")
using .Data
export get_train_data, get_x2, create_inputs, split_data, get_screen_norm, get_max_norm, convert_coords_to_clusters, plot_convert_coords, denorm, soft_denorm, MeanNorm, MeanIntNorm, ClusterNorm, MixBatchNorm, apply_norm, merge_norms, MixBatchNorm

include("draw.jl")
using .Draw
export append_to_csv, load_image_and_draw, draw_linewidth!, scale2pixX2, PointRound, PixelPoint, scale2pix

include("utils.jl")
using .Utils
export get_file_timestamps, get_video_duration, get_fps, extract_all_frames, get_start_date_from_birth, get_start_date_from_modify, extract_date_from_path, cutvid_at_sec, get_paths, run_in_shell

include("model.jl")
using .Model
export create_model

include("train.jl")
using .Train
export train_model!

end # module AutoGamer
