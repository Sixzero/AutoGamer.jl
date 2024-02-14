module Utils

using Dates
using Glob
using Pipe

export get_file_timestamps, get_video_duration, get_fps, extract_all_frames, get_start_date_from_birth, get_start_date_from_modify, extract_date_from_path, cutvid_at_sec, get_paths, run_in_shell
# parsing a date like: 2024-01-13 17:29:43.299423030
function parse_stat_date(date_str)
  datetime_part, fractional_part = split(date_str, ".")

  date = DateTime(datetime_part, "yyyy-mm-dd HH:MM:SS")
  millis = parse(Int, fractional_part[1:3])
  # combine the date and the millisecond part
  return date + Millisecond(millis)
end
function get_file_timestamps(file_path::String)
  stat_output = read(`stat $file_path`, String)
  birth = modify = ""
  for line in split(stat_output, '\n')
    if startswith(line, " Birth:")
        birth = strip(join(split(line, " ")[3:4], " "))
    elseif startswith(line, "Modify:")
        modify = strip(join(split(line, " ")[2:3], " "))
    end
  end

  birth_date = parse_stat_date(birth)
  modify_date = parse_stat_date(modify)
  return birth_date, modify_date
end
run_in_shell(shellcmd) = run(`sh -c $shellcmd`)

function get_video_duration(file_path::String)
  cmd = `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file_path`
  ffmpeg_output = read(cmd, String)
  return parse(Float32, ffmpeg_output)
end
function get_fps(video_path)
  fps_str = read(`ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 $video_path`, String)
  # value of num/den so we need to split and convert to real
  fps = (sp = split(fps_str, "/"); parse(Float32, sp[1]) / parse(Float32, sp[2]))
end
function extract_all_frames(video_path, fps=get_fps(video_path))
  folder = @pipe split(video_path, "/")[1:end-1] |> join(_, "/")
  file_date = extract_date_from_path(video_path)
  @info `ffmpeg -i $video_path -vf "fps=$(fps),scale=108:108" $folder/extracted/$(file_date)_%d.png`
  run(`ffmpeg -i $video_path -vf "fps=$(fps),scale=108:108" $folder/extracted/$(file_date)_%d.png`)
end
get_start_date_from_birth(f_path) = get_file_timestamps(f_path)[1]
function get_start_date_from_modify(f_path)
  path_date = extract_date_from_path(f_path)
  @show path_date
  birth, modify = get_file_timestamps(f_path)
  @show birth
  @show modify
  # @show birth
  # @show modify
  duration = get_video_duration(f_path)
  @show duration
  calculated = modify - Millisecond(duration*1000)
  @show calculated
  MAX_DIFF = Millisecond(1000)
  @assert -MAX_DIFF < calculated - path_date < MAX_DIFF "The difference between birth and modify is too big: $(calculated - path_date)"
  return calculated
end
function extract_date_from_path(path="data/alien/rec_2024-01-03_19-51-31.mkv")
  path = split(path, "/")[end]
  date_part = path[5:end-4]
  date_parsed = Dates.DateTime(date_part, )
  date_parsed
end
function cutvid_at_sec(cut_at_sec, video_path, output_filename, overwrite=false)
  if overwrite  
    run(`ffmpeg -y -loglevel error -ss $cut_at_sec -i $video_path -frames:v 1 $output_filename`)
  else
    run(`ffmpeg -loglevel error -ss $cut_at_sec -i $video_path -frames:v 1 $output_filename`)
  end
end

function get_paths(date, game)

  # find files which start with video_file 
  video_files = glob("data/$game/rec_$(date)*.mkv")
  @assert length(video_files) == 1 "The code is only prepared for 1 match. We have: $(length(video_files)) matches."
  video_file = video_files[1]
  touch_file = "data/$game/touch_events_$(date).csv"
  return video_file, touch_file
end

end # module