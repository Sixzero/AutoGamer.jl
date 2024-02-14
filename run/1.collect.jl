using Printf
using Dates
using CSV
using DataFrames
using AutoGamer.Draw

using Boilerplate: @asyncsafe

# Event Types
const EV_SYN = 0x00
const EV_ABS = 0x03
const EV_KEY = 0x01

# ABS Event Codes
const ABS_X = 0x35
const ABS_Y = 0x36
const ABS_MT_POSITION_X = 0x35  # Multitouch X position
const ABS_MT_POSITION_Y = 0x36  # Multitouch Y position

# KEY Event Codes
const BTN_TOUCH = 0x14a


current_time = "$(now())"
function parse_event_line(line::String)
    parts = split(line)
    if length(parts) == 4 && parts[1] == "/dev/input/event2:"
        event_type = parse(Int, parts[2], base=16)
        event_code = parse(Int, parts[3], base=16)
        value = parse(Int, parts[4], base=16)
        return event_type, event_code, value
    end
    return nothing
end
function start_recording(game)
    println("Recording Start Time: ", current_time)
    # process = @asyncsafe run(`scrcpy --record rec_$current_time.mkv`, wait=false)
    process = open(`scrcpy --record data/$game/rec_$current_time.mkv`, "r")
end
function capture_touch_events(game)
  adb_command = `adb shell getevent`
  process = open(adb_command, "r")

  touch_active = false
  x = y = 0
  start_x = start_y = 0
  touch_events = DataFrame(Timestamp = DateTime[], X = Int[], Y = Int[], Start_X = Int[], Start_Y = Int[])
  csv_file_path = "data/$game/touch_events_$current_time.csv"

  @asyncsafe for line in eachline(process)
      result = parse_event_line(line)
      if result !== nothing
          event_type, event_code, value = result

          if event_type == EV_ABS
              if event_code in [ABS_X, ABS_MT_POSITION_X]
                  x = value
              elseif event_code in [ABS_Y, ABS_MT_POSITION_Y]
                  y = value
              end
          elseif event_type == EV_KEY && event_code == BTN_TOUCH
              if value != 0
                  touch_active = true
                  start_x, start_y = x, y
              else
                  touch_active = false
              end
          elseif event_type == EV_SYN && touch_active
              timestamp = now()
            #   println("[$timestamp] Touch at position: X=$x, Y=$y, Start X=$start_x, Start Y=$start_y")
              push!(touch_events, (timestamp, x, y, start_x, start_y))
              touch_event = DataFrame(timestamp = [timestamp], x = [x], y = [y], start_x = [start_x], start_y = [start_y])
              @time append_to_csv(csv_file_path, touch_event)
          end
      end
  end
  process
end

function main(game)
    local p1, p2
    try
        p1 = start_recording(game)
        p2 = capture_touch_events(game)
        wait(p1); wait(p2)
    catch e
        if isa(e, InterruptException)
            println("Interrupted by user")
            kill(p1); kill(p2)
        else
            rethrow(e)
        end
    end
end

game = "survivor"
game = "alien"
main(game)
#%%