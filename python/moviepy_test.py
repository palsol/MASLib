from moviepy.editor import *

video = VideoFileClip("/home/palsol/CLionProjects/MASLib/data/res/animate/M O O N - Crystals.mp4").subclip(0, 1)

# Make the text. Many more options are available.
txt_clip = (TextClip("My Holidays 2013", fontsize=70, color='white')
            .set_duration(1))

result = CompositeVideoClip([video, txt_clip])  # Overlay text on video
result.write_videofile("./myHolidays_edited.webm", fps=25)  # Many options...
