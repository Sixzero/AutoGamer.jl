
#%%
using ImageDraw
using ImageView, Images
for i in 1:20
	@sizes frms = get_next_frames(video_socket, 0x2000)
	if frms === nothing || length(frms)==0
		continue
	end
	img = RGB{N0f8}.(frms[end]./255)
	@sizes img
	@display img
	break
end
;
#%%

#%%
# Parse packets using av
packets = codec[:parse]((raw_h264))
@typeof packets

if isempty(packets)
		return nothing
end

result_frames = []

# Decode each packet to frames
for packet in packets
	@typeof packet
	@typeof av.packet.Packet(packet)
		frames = codec[:decode](av.packet.Packet(packet))
		for frame in frames
				# Convert frame to ndarray in RGB24 format
				rgb_frame = frame[:to_ndarray](format="rgb24")
				push!(result_frames, rgb_frame)
		end
end

#%%
function process_h264_to_rgb_images(tmp_filename)
	# Define the output pattern for the image files
	output_pattern = "frame_%04d.png"  # This will create images like frame_0001.png, frame_0002.png, etc.

	# Construct the ffmpeg command
	# ffmpeg_command = `ffmpeg -v error -i $tmp_filename -f image2 -vcodec png -pix_fmt rgb24 frame_%04d.png`

	ffmpeg_command = `ffmpeg -v error -err_detect ignore_err -i $tmp_filename -f image2 -vcodec png -pix_fmt rgb24 $output_pattern`

	# Execute the ffmpeg command
	run(ffmpeg_command)

	# You might want to return the list of generated image filenames or handle them in some way
end

while true
	# Read raw H264 data from the socket
	@time raw_h264 = read(video_socket, 0x10000)

	# Save raw H264 data to a temporary file
	tmp_filename = tempname() * ".h264"
	open(tmp_filename, "w") do f
			write(f, raw_h264)
	end
	@show tmp_filename

	process_h264_to_rgb_images(tmp_filename)
	
end
#%%
frames = get_next_frames(video_socket)
using Boilerplate
@sizes frames
#%%
@typeof res
Int.(res)