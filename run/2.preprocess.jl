
using CSV
using DataFrames
using Dates
using AutoGamer
# extract frame every 5 seconds
`ffmpeg -i input.mp4 -vf "fps=1/5" output_%d.png`

function get_output_for_timestamp(ts::DateTime, data::DataFrame)
    # in julia find first data row in data which comes after ts
    idx::Int = 0
    timestamps::Vector{DateTime} = data.timestamp
    for (i, data_ts) in enumerate(timestamps)
        if ts <= data_ts
            idx = i
            break
        end
    end
    if idx == 0
        return nothing
    end
    if timestamps[idx] - ts > Millisecond(40) 
        # println("Too far. $(data[idx, :timestamp] - ts)")
        return nothing
    end
    x::Int,y::Int, x_s::Int,y_s::Int= data[idx, [:x,:y, :start_x, :start_y]]
    return scale2pixX2(x, y, x_s, y_s)
end
function register_data(png_file, ts, touch)
    global registered_data_df
    # check whether ts is already in the csv
    # registered_data_df = isfile(csv_file_path) && CSV.read(csv_file_path, DataFrame)
    if isfile(csv_file_path) && (ts in registered_data_df[!, :timestamp])
        return
    end
    
    local train_data
    if touch === nothing
        train_data = DataFrame(timestamp=[ts], file=[png_file], x = [-1], y = [-1], start_x = [-1], start_y = [-1])
    else
        x,y,x_s,y_s = touch
        train_data = DataFrame(timestamp=[ts], file=[png_file], x = [x], y = [y], start_x = [x_s], start_y = [y_s])
    end
    append_to_csv(csv_file_path, train_data)
    # append data to registered_data_df
    registered_data_df = vcat(registered_data_df, train_data)
end

game = "alien"
csv_file_path = "data/$game/data_map.csv"
output_dir = "data/$game/extracted"

# "2024-01-04T13:25:14.109", 
# date = "2024-01-16T19:00:23.784"
good_dates = ["2024-01-29T11:33:53.241", "2024-02-04T11:38:44.227"]
date = good_dates[end]
cut_fps = 15
step_size = 1/cut_fps # get_fps(vp)
# vp, tp = get_paths(date, game)
# start_ts = datetime2unix(get_start_date_from_modify(vp))
# extract_all_frames(vp, cut_fps)
#%%
using CSV
run_in_shell("rm -rf $output_dir\\*")
`rm -rf $csv_file_path ` |> run
run_in_shell("touch $csv_file_path")
run_in_shell("chmod 777 $csv_file_path")
run_in_shell("echo \"timestamp,file,x,y,start_x,start_y\" > $(csv_file_path)")

registered_data_df = CSV.read(csv_file_path, DataFrame)
@show registered_data_df
function match_touches_with_frames(date, game, step_size)
    vp, tp = get_paths(date, game)
    extract_all_frames(vp, cut_fps)
    data = CSV.read(tp, DataFrame)
    start_ts = datetime2unix(get_start_date_from_modify(vp))
    file_date = extract_date_from_path(vp)
    video_duration = get_video_duration(vp)
    cut_frames = floor(Int, video_duration / step_size) - 1
    no_data_count = 0
    for i in 1:cut_frames
        cut_ts = i * step_size
        # @show cut_ts
        timestamp = unix2datetime(cut_ts + start_ts)
        output_filename = joinpath(output_dir, "$(file_date)_$i.png")
        coords = get_output_for_timestamp(timestamp, data) # unconverted Tuple(v for v in row
        if coords == nothing
            no_data_count += 1
            println("$no_data_count/$i the reatio in nodata vs data")
        end
        register_data(output_filename, timestamp, coords)

    end
end
for date in good_dates
    match_touches_with_frames(date, game, step_size)
end
;
#%%

