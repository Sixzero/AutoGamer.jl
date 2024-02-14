module Draw

using Images
using ImageDraw
using DataFrames
using CSV

export append_to_csv, load_image_and_draw, draw_linewidth!, scale2pixX2, PointRound, PixelPoint, scale2pix
function append_to_csv(file_path::String, data::DataFrame)
  CSV.write(file_path, data, append=true, header=!isfile(file_path))
end
PointRound(x,y) = ImageDraw.Point(round(Int, x), round(Int, y))
PixelPoint(x,y) = begin # also round to int
  PointRound(x/8615*1080, y/19175*2400)
end
function scale2pix(x,y) 
  x/8615*1080, y/19175*2400
end
scale2pixX2(x,y,x2,y2) = scale2pix(x,y)..., scale2pix(x2,y2)...

load_image_and_draw(png_file, line::Nothing) = false
function load_image_and_draw(png_file, line)
    x,y,x_s,y_s = line
    @show x,y,x_s,y_s
    img = load(png_file)
    @show size(img)
    draw_linewidth!(img, LineSegment(PointRound(x, y), PointRound(x_s, y_s)), RGB{N0f8}(1,0,0), 45)
    display(img)
    # save(img, png_file)
    true
end


draw_linewidth!(img, line::LineSegment, color, width=5) = begin
  dx = line.p2.x - line.p1.x
  dy = line.p2.y - line.p1.y
  len = sqrt(dx^2 + dy^2)

  if len == 0
    # Draw a dot at line.p1 with the specified width
    draw_dot!(img, line.p1, color, width)  # Assuming draw_dot! is your dot drawing function
    return
  end
  ux, uy = dy / len, -dx / len  # Unit vector perpendicular to the line

  half_width = width / 2
  for offset in -half_width:half_width
      new_p1 = (line.p1.x + offset * ux, line.p1.y + offset * uy)
      new_p2 = (line.p2.x + offset * ux, line.p2.y + offset * uy)
      new_line = LineSegment(PointRound(new_p1...), PointRound(new_p2...))
      draw!(img, new_line, color)
  end
  # @show line.p1.x, line.p1.y, line.p2
  # for i in 1:10:width
  #     draw!(img, line, color)
  # end
end 


function draw_dot!(img, center::ImageDraw.Point, color, dot_size)
  num_lines = 150  # Number of lines to draw (like a star)
  radius = dot_size * 2  # Radius of the dot
  width, height = size(img)
  # for i in 1:num_lines
  #     angle = 2 * π * i / num_lines
  #     end_x = round(Int, center.x + radius * cos(angle))
  #     end_y = round(Int, center.y + radius * sin(angle))
  #     line = LineSegment(center, ImageDraw.Point(end_x, end_y))
  #     draw!(img, line, color)
  # end

  for dx in -dot_size:dot_size
    for dy in -dot_size:dot_size
        x, y = center.x + dx, center.y + dy

        # Check if the point is inside the image bounds
        if x > 0 && x <= width && y > 0 && y <= height
            # Check if the point is inside the circle
            if dx^2 + dy^2 <= dot_size*dot_size
                img[x, y] = (.5*color + 0.5*img[x, y]) # Set the color
            end
        end
    end
end
end

end # module Draw