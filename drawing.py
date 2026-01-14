import pygame
import math
import numpy as np
import os
import subprocess

# Color definitions
bg_color = (22, 26, 35)
ground_color = (35, 38, 42)
ground_edge = (50, 55, 60)
robot_body = (50, 220, 235)
robot_joint = (255, 120, 180)
robot_vertex = (80, 240, 255)
door_color = (80, 255, 120)
text_white = (220, 220, 230)
yellow = (255, 230, 100)

# Constants
WIDTH, HEIGHT = 800, 600

def draw_gradient_background(screen, width, height):
    """Draw background."""
    screen.fill(bg_color)

def draw_thick_line_as_polygon(screen, color, start, end, thickness):
    """
    Draw a thick line as a polygon.
    
    Args:
        screen: Pygame screen surface
        color: Color tuple (R, G, B)
        start: Start point tuple (x, y)
        end: End point tuple (x, y)
        thickness: Line thickness in pixels
    """
    # Calculate the angle between the start and end points
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # Calculate the offset from the start point based on the angle and thickness
    dx = thickness / 2 * math.sin(angle)
    dy = thickness / 2 * math.cos(angle)

    # Define the points of the polygon
    points = [
        (start[0] - dx, start[1] + dy),
        (start[0] + dx, start[1] - dy),
        (end[0] + dx, end[1] - dy),
        (end[0] - dx, end[1] + dy)
    ]

    pygame.draw.polygon(screen, color, points)

def draw_door(screen, door_x, door_w, door_h):
    """Draw door with radial glow effect and frame."""
    top_left_x = float(door_x) - float(door_w) / 2
    top_left_y = HEIGHT - float(door_h)
    door_w_f = float(door_w)
    door_h_f = float(door_h)
    
    door_center_x = int(door_x)
    door_center_y = int(HEIGHT - door_h_f / 2)
    
    max_glow_radius = 80
    glow_layers = 50
    
    for i in range(glow_layers, 0, -1):
        t = i / glow_layers
        radius_x = int(max_glow_radius * t + door_w_f / 2)
        radius_y = int(max_glow_radius * t * 0.7 + door_h_f / 2)
        fade = (1 - t) ** 0.6
        inner_t = 1 - t
        ease = inner_t * inner_t * (3 - 2 * inner_t)
        
        glow_r = 35 + 25 * ease
        glow_g = 50 + 40 * ease
        glow_b = 45 + 15 * ease
        
        bg_r, bg_g, bg_b = bg_color
        intensity = fade * 0.25
        r = bg_r + (glow_r - bg_r) * intensity
        g = bg_g + (glow_g - bg_g) * intensity
        b = bg_b + (glow_b - bg_b) * intensity
        
        glow_rect = pygame.Rect(
            door_center_x - radius_x,
            door_center_y - radius_y,
            radius_x * 2,
            radius_y * 2
        )
        pygame.draw.ellipse(screen, (int(r), int(g), int(b)), glow_rect)
    
    door_interior = (35, 90, 55)
    pygame.draw.rect(screen, door_interior, 
                     (int(top_left_x), int(top_left_y), int(door_w_f), int(door_h_f)))
    
    frame_color = (100, 255, 140)
    frame_width = 3
    pygame.draw.rect(screen, frame_color, 
                     (int(top_left_x), int(top_left_y), int(door_w_f), int(door_h_f)), 
                     frame_width)

def draw_ground_from_points(screen, height, points):
    """Draw ground from saved points."""
    for i in range(len(points) - 1):
        a = (int(float(points[i][0])), int(height - float(points[i][1])))
        b = (int(float(points[i+1][0])), int(height - float(points[i+1][1])))
        ground_poly = [a, b, (int(b[0]), int(height)), (int(a[0]), int(height))]
        pygame.draw.polygon(screen, ground_color, ground_poly)
        pygame.draw.line(screen, ground_edge, a, b, 2)

def bevel_polygon(vertices, bevel_size=4):
    """Convert polygon to beveled version with chamfered corners."""
    n = len(vertices)
    beveled = []
    
    for i in range(n):
        prev_v = vertices[(i - 1) % n]
        curr_v = vertices[i]
        next_v = vertices[(i + 1) % n]
        
        to_prev = (prev_v[0] - curr_v[0], prev_v[1] - curr_v[1])
        to_next = (next_v[0] - curr_v[0], next_v[1] - curr_v[1])
        
        len_prev = math.sqrt(to_prev[0]**2 + to_prev[1]**2)
        len_next = math.sqrt(to_next[0]**2 + to_next[1]**2)
        
        if len_prev > 0 and len_next > 0:
            unit_prev = (to_prev[0] / len_prev, to_prev[1] / len_prev)
            unit_next = (to_next[0] / len_next, to_next[1] / len_next)
            actual_bevel = min(bevel_size, len_prev * 0.4, len_next * 0.4)
            
            p1 = (curr_v[0] + unit_prev[0] * actual_bevel,
                  curr_v[1] + unit_prev[1] * actual_bevel)
            p2 = (curr_v[0] + unit_next[0] * actual_bevel,
                  curr_v[1] + unit_next[1] * actual_bevel)
            
            beveled.append((int(p1[0]), int(p1[1])))
            beveled.append((int(p2[0]), int(p2[1])))
        else:
            beveled.append((int(curr_v[0]), int(curr_v[1])))
    
    return beveled

def draw_robot_from_vertices(screen, height, vertices1_world, vertices2_world, joint_pos):
    """Draw robot with beveled corners."""
    vertices1_screen = [(float(v[0]), height - float(v[1])) for v in vertices1_world]
    vertices2_screen = [(float(v[0]), height - float(v[1])) for v in vertices2_world]
    joint_screen = (int(float(joint_pos[0])), int(height - float(joint_pos[1])))
    
    bevel_amount = 5
    beveled1 = bevel_polygon(vertices1_screen, bevel_amount)
    beveled2 = bevel_polygon(vertices2_screen, bevel_amount)
    
    pygame.draw.polygon(screen, robot_body, beveled1)
    pygame.draw.polygon(screen, robot_body, beveled2)
    pygame.draw.polygon(screen, robot_vertex, beveled1, 2)
    pygame.draw.polygon(screen, robot_vertex, beveled2, 2)
    
    pygame.draw.circle(screen, robot_joint, joint_screen, 7)
    pygame.draw.circle(screen, (255, 180, 210), (joint_screen[0] - 2, joint_screen[1] - 2), 3)

def draw_robot_ghost(screen, height, vertices1_world, vertices2_world, joint_pos, fade):
    """
    Draw a faded ghost of the robot for trail effect.
    
    Args:
        fade: 0.0 = fully faded (invisible), 1.0 = fully visible
    """
    vertices1_screen = [(float(v[0]), height - float(v[1])) for v in vertices1_world]
    vertices2_screen = [(float(v[0]), height - float(v[1])) for v in vertices2_world]
    joint_screen = (int(float(joint_pos[0])), int(height - float(joint_pos[1])))
    
    # Create beveled shapes
    bevel_amount = 5
    beveled1 = bevel_polygon(vertices1_screen, bevel_amount)
    beveled2 = bevel_polygon(vertices2_screen, bevel_amount)
    
    # Fade colors toward background
    bg = bg_color
    ghost_body = (
        int(bg[0] + (robot_body[0] - bg[0]) * fade * 0.4),
        int(bg[1] + (robot_body[1] - bg[1]) * fade * 0.4),
        int(bg[2] + (robot_body[2] - bg[2]) * fade * 0.4)
    )
    ghost_joint = (
        int(bg[0] + (robot_joint[0] - bg[0]) * fade * 0.3),
        int(bg[1] + (robot_joint[1] - bg[1]) * fade * 0.3),
        int(bg[2] + (robot_joint[2] - bg[2]) * fade * 0.3)
    )
    
    # Draw faded robot
    pygame.draw.polygon(screen, ghost_body, beveled1)
    pygame.draw.polygon(screen, ghost_body, beveled2)
    pygame.draw.circle(screen, ghost_joint, joint_screen, 5)

def draw_robot_with_trail(screen, height, history, trail_length=4):
    """Draw robot with motion trail (ghost frames)."""
    if not history:
        return
    
    for i in range(min(trail_length, len(history) - 1)):
        history_idx = len(history) - 2 - i
        if history_idx >= 0:
            vertices1, vertices2, joint_pos = history[history_idx]
            fade = 1.0 - (i + 1) / (trail_length + 1)
            draw_robot_ghost(screen, height, vertices1, vertices2, joint_pos, fade)
    
    vertices1, vertices2, joint_pos = history[-1]
    draw_robot_from_vertices(screen, height, vertices1, vertices2, joint_pos)

def parse_step_data(step_data):
    """Extract action, vertices, joint position, and cumulative return from step data."""
    action = step_data[0]
    vertices1_flat = step_data[1:9]
    vertices1_world = [(vertices1_flat[i], vertices1_flat[i+1]) for i in range(0, 8, 2)]
    vertices2_flat = step_data[9:17]
    vertices2_world = [(vertices2_flat[i], vertices2_flat[i+1]) for i in range(0, 8, 2)]
    joint_pos = (float(step_data[17]), float(step_data[18]))
    cumulative_return = float(step_data[19]) if len(step_data) > 19 else None
    return action, vertices1_world, vertices2_world, joint_pos, cumulative_return

def draw_torque_bar(screen, font, action, crop_offset=0):
    """
    Draw a horizontal torque bar showing action value (-1 to 1).
    Positive (right) = light blue, Negative (left) = light pink/red.
    Top right corner, with label above.
    
    Args:
        crop_offset: Y offset for 16:9 crop mode (150 when cropping)
    """
    # Bar dimensions - horizontal, smaller
    bar_width = 80
    bar_height = 8
    bar_x = 705  # Right side with margin
    bar_y = crop_offset + 25  # Top area (offset for crop mode)
    
    # Colors
    bar_bg = (40, 42, 48)  # Dark background for bar
    bar_border = (60, 65, 75)  # Subtle border
    positive_color = (120, 180, 220)  # Light blue
    negative_color = (220, 130, 140)  # Light pink/red
    center_line = (80, 85, 95)  # Center marker
    label_color = (160, 165, 175)  # Muted text
    
    # Draw "Torque" label above bar
    label = font.render("Torque", True, label_color)
    label_x = bar_x + bar_width // 2 - label.get_width() // 2
    label_y = bar_y - 16
    screen.blit(label, (label_x, label_y))
    
    # Draw bar background
    pygame.draw.rect(screen, bar_bg, (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, bar_border, (bar_x, bar_y, bar_width, bar_height), 1)
    
    # Draw center line (zero point)
    center_x = bar_x + bar_width // 2
    pygame.draw.line(screen, center_line, (center_x, bar_y - 2), (center_x, bar_y + bar_height + 2), 1)
    
    # Clamp action to -1, 1 range
    action_clamped = max(-1.0, min(1.0, float(action)))
    
    # Calculate fill width (half bar width = full range on each side)
    half_width = bar_width // 2
    fill_width = int(abs(action_clamped) * half_width)
    
    if fill_width > 0:
        if action_clamped > 0:
            # Positive: fill rightward from center (light blue)
            fill_rect = (center_x, bar_y + 1, fill_width, bar_height - 2)
            pygame.draw.rect(screen, positive_color, fill_rect)
        else:
            # Negative: fill leftward from center (light pink)
            fill_rect = (center_x - fill_width, bar_y + 1, fill_width, bar_height - 2)
            pygame.draw.rect(screen, negative_color, fill_rect)

def draw_time_left(screen, font, time_left, crop_offset=0):
    """Draw time remaining display."""
    label_color = (160, 165, 175)
    time_color = (200, 205, 215)
    x_pos = 705
    y_pos = crop_offset + 50
    
    label = font.render("Time Left", True, label_color)
    screen.blit(label, (x_pos, y_pos))
    
    time_str = f"{time_left:.1f}s"
    time_text = font.render(time_str, True, time_color)
    time_x = x_pos + label.get_width() // 2 - time_text.get_width() // 2
    screen.blit(time_text, (time_x, y_pos + 18))

def draw_episode_number(screen, font, episode_num, crop_offset=0):
    """Draw episode number display."""
    label_color = (160, 165, 175)
    value_color = (200, 205, 215)
    x_pos = 705
    y_pos = crop_offset + 90
    
    label = font.render("Episode", True, label_color)
    screen.blit(label, (x_pos, y_pos))
    
    value_text = font.render(str(episode_num), True, value_color)
    value_x = x_pos + label.get_width() // 2 - value_text.get_width() // 2
    screen.blit(value_text, (value_x, y_pos + 18))

def replay_episode(episode_file, speed=1.0, time_limit=20.0, fps=30):
    """
    Replay a saved episode in a pygame window.
    Controls: SPACE (pause/unpause), ESC (exit), R (replay after completion).
    """
    # Load episode data
    if not os.path.exists(episode_file):
        raise FileNotFoundError(f"Episode file not found: {episode_file}")
    
    data = np.load(episode_file)
    points = data['points']
    door = data['door']
    steps = data['steps']
    ep_return = float(data['ep_return'])
    
    # Set up pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Episode Replay - Return: {ep_return:.1f}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Parse door
    door_x, door_w, door_h = door
    
    running = True
    paused = False
    step_idx = 0
    completed = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r and completed:
                    # Replay from beginning
                    step_idx = 0
                    completed = False
                    paused = False
        
        if not completed:
            if step_idx >= len(steps):
                completed = True
            else:
                # Parse step data
                action, vertices1_world, vertices2_world, joint_pos, cumulative_return = parse_step_data(steps[step_idx])
                
                # Calculate time left
                current_time = step_idx / fps
                time_left = max(0, time_limit - current_time)
                
                # Draw everything
                draw_gradient_background(screen, WIDTH, HEIGHT)
                draw_door(screen, door_x, door_w, door_h)
                draw_ground_from_points(screen, HEIGHT, points)
                draw_robot_from_vertices(screen, HEIGHT, vertices1_world, vertices2_world, joint_pos)
                draw_torque_bar(screen, font, action)
                draw_time_left(screen, font, time_left)
                
                # Show pause indicator
                if paused:
                    pause_msg = "PAUSED - Press SPACE to resume"
                    pause_txt = font.render(pause_msg, True, yellow)
                    screen.blit(pause_txt, (10, 40))
                
                pygame.display.flip()
                
                # Only advance if not paused
                if not paused:
                    clock.tick(int(fps * speed))
                    step_idx += 1
                else:
                    clock.tick(fps)  # Still tick when paused for responsiveness
        else:
            # Show completion message
            draw_gradient_background(screen, WIDTH, HEIGHT)
            draw_door(screen, door_x, door_w, door_h)
            draw_ground_from_points(screen, HEIGHT, points)
            # Draw last frame
            if len(steps) > 0:
                action, vertices1_world, vertices2_world, joint_pos, cumulative_return = parse_step_data(steps[-1])
                draw_robot_from_vertices(screen, HEIGHT, vertices1_world, vertices2_world, joint_pos)
            
            msg = "Replay complete. Press R to replay, ESC to exit"
            txt = font.render(msg, True, yellow)
            screen.blit(txt, (10, 40))
            pygame.display.flip()
            clock.tick(10)
    
    pygame.quit()

def render_episode_to_video(episode_file, output_video, fps=30, speed=1.0, 
        width=WIDTH, height=HEIGHT, crop_16_9=False, time_limit=20.0, episode_num=None):
    """Render a saved episode to video file using FFmpeg."""
    # Load episode data
    if not os.path.exists(episode_file):
        raise FileNotFoundError(f"Episode file not found: {episode_file}")
    
    data = np.load(episode_file)
    points = data['points']
    door = data['door']
    steps = data['steps']
    ep_return = float(data['ep_return'])
    
    # Parse door
    door_x, door_w, door_h = door
    
    if episode_num is None:
        basename = os.path.basename(episode_file)
        try:
            episode_num = int(basename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            episode_num = 0
    
    output_fps = int(fps * speed)
    if output_fps < 1:
        output_fps = 1
    
    if crop_16_9:
        crop_height = int(width * 9 / 16)
        crop_offset = height - crop_height
    else:
        crop_offset = 0
    
    pygame.init()
    screen = pygame.Surface((width, height))
    font = pygame.font.SysFont(None, 24)
    
    # Set up FFmpeg subprocess
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'rawvideo',
        '-pixel_format', 'rgb24',
        '-video_size', f'{width}x{height}',
        '-framerate', str(output_fps),
        '-i', '-',  # Read from stdin
    ]
    
    # Add crop filter if requested
    if crop_16_9:
        crop_height = int(width * 9 / 16)
        ffmpeg_cmd.extend(['-vf', f'crop={width}:{crop_height}:0:{crop_offset}'])
    
    ffmpeg_cmd.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        output_video
    ])
    
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Check if episode ended early (robot reached door) - ended before max frames
        max_frames = int(time_limit * fps)
        episode_success = len(steps) < max_frames and ep_return > 0
        
        # Pause frames at end for successful episodes (0.5 seconds)
        pause_frames = int(fps * 0.5) if episode_success else 0
        total_frames = len(steps) + pause_frames
        
        crop_info = " (16:9 crop)" if crop_16_9 else ""
        success_info = " [SUCCESS - adding pause]" if episode_success else ""
        print(f"Rendering {total_frames} frames to {output_video} at {output_fps} fps (speed: {speed}x){crop_info}{success_info}...")
        
        # Render each frame
        last_frame_bytes = None
        for step_idx in range(len(steps)):
            # Parse step data
            action, vertices1_world, vertices2_world, joint_pos, cumulative_return = parse_step_data(steps[step_idx])
            
            # Calculate time left
            current_time = step_idx / fps
            time_left = max(0, time_limit - current_time)
            
            # Draw everything to surface
            draw_gradient_background(screen, width, height)
            draw_door(screen, door_x, door_w, door_h)
            draw_ground_from_points(screen, height, points)
            draw_robot_from_vertices(screen, height, vertices1_world, vertices2_world, joint_pos)
            draw_torque_bar(screen, font, action, crop_offset)
            draw_time_left(screen, font, time_left, crop_offset)
            draw_episode_number(screen, font, episode_num, crop_offset)
            
            # Convert pygame surface to raw RGB bytes
            # pygame.surfarray.array3d returns (height, width, 3) array
            frame_array = pygame.surfarray.array3d(screen)
            # Transpose to (width, height, 3) and convert to uint8
            frame_array = np.transpose(frame_array, (1, 0, 2)).astype(np.uint8)
            # Convert to raw bytes (row-major order)
            frame_bytes = frame_array.tobytes()
            last_frame_bytes = frame_bytes  # Save for pause
            
            # Write frame to FFmpeg stdin
            ffmpeg_process.stdin.write(frame_bytes)

            if (step_idx + 1) % 50 == 0 or step_idx == len(steps) - 1:
                progress = (step_idx + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% ({step_idx + 1}/{total_frames} frames)")
        
        if pause_frames > 0 and last_frame_bytes:
            for i in range(pause_frames):
                ffmpeg_process.stdin.write(last_frame_bytes)
            print(f"Progress: 100.0% ({total_frames}/{total_frames} frames) - pause added")
        
        ffmpeg_process.stdin.close()
        stdout, stderr = ffmpeg_process.communicate()
        
        if ffmpeg_process.returncode == 0:
            print(f"Video saved successfully to {output_video}")
        else:
            print(f"FFmpeg error: {stderr.decode()}")
            raise RuntimeError(f"FFmpeg failed with return code {ffmpeg_process.returncode}")
    
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
    except Exception as e:
        # Clean up on error
        if 'ffmpeg_process' in locals():
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
        raise e
    finally:
        pygame.quit()

if __name__ == "__main__":
    # Example: Render all episodes to videos
    episodes_dir = "runs/episodes"
    videos_dir = "runs/videos"
    os.makedirs(videos_dir, exist_ok=True)

    episode_files = sorted([
        f for f in os.listdir(episodes_dir)
        if f.endswith(".npz")
    ])

    for ep_file in episode_files:
        ep_path = os.path.join(episodes_dir, ep_file)
        video_name = ep_file.replace(".npz", ".mp4")
        video_path = os.path.join(videos_dir, video_name)
        print(f"Rendering {ep_path} -> {video_path}")
        try:
            render_episode_to_video(ep_path, video_path, speed=1, crop_16_9=True)
        except Exception as e:
            print(f"Failed to render {ep_file}: {e}")


    # Example: Replay an episode
    episode_file = "runs/episodes/ep_000000.npz"
    replay_episode(episode_file, speed=1.0, time_limit=20.0, fps=30)

