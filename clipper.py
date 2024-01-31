from langchain_community.llms.gpt4all import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import whisper
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import re
import assemblyai as aai

video_name = "input_3.mp4" # you can change that to your desired video

aai.settings.api_key = "b957ecf80fe24b5c907dcd7199b80f9e"

def ms_to_hms(start):
    s, ms = divmod(start, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s

def create_timestamps(chapters):
    last_hour = ms_to_hms(chapters[-1].start)[0]
    time_format = "{m:02d}:{s:02d}" if last_hour == 0 else "{h:02d}:{m:02d}:{s:02d}"

    lines = []
    for idx, chapter in enumerate(chapters):
        h, m, s = (0, 0, 0) if idx == 0 else ms_to_hms(chapter.start)
        lines.append(f"{time_format.format(h=h, m=m, s=s)} {chapter.headline}")
    return "\n".join(lines)

# we add a TranscriptionConfig to turn on Auto Chapters
transcriber = aai.Transcriber(
  config=aai.TranscriptionConfig(auto_chapters=True)
)

transcript = transcriber.transcribe(video_name)

if transcript.error: raise RuntimeError(transcript.error)

# print the text
print(transcript.text, end='\n\n')

# now we print the video sections information
for chapter in transcript.chapters:
    print(f"Start: {chapter.start}, End: {chapter.end}")
    print(f"Summary: {chapter.summary}")
    print(f"Healine: {chapter.headline}")
    print(f"Gist: {chapter.gist}")

timestamp_lines = create_timestamps(transcript.chapters)
print(timestamp_lines)

with open("chapters.txt", "w") as f:
    f.write(timestamp_lines+"\n")

model = whisper.load_model("base")
video = mp.VideoFileClip(video_name)

PATH = 'C:/Users/ammar/AppData/Local/nomic.ai/GPT4All/mistral-7b-openorca.Q4_0.gguf'
llm = GPT4All(model=PATH, n_threads=8)

chapters = ""
chap_list = []
timelines = []

with open("chapters.txt", "r") as f:
    file = f.readlines()
    for i in file:
        chapters += i
        chap_list.append(i.split()[0])
print(chap_list)

prompt = PromptTemplate(
    input_variables=["input"],
    template='''
    Question: {input}
    Answer: go through all the timeframes and give me the ones that answers the question 
    '''
)


# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "When answering any user question related to the video, write the output in this format:\nTimeframe + text\ntimeframe"),
#     ("user", "{input}")
# ])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

try:
    output = chain.invoke({"input": f"{chapters}\nwhat does elon musk think would happen in the future"})
except Exception as e:
    print(f"An error occurred: {e}")
else:
    for i in output.split():
        if re.match('^[0-9]{2}\:[0-9]{2}\:[0-9]{2}$',i):
            ind = chap_list.index(i)
            start = [int(i) for i in chap_list[ind].split(":")]
            end = [int(i) for i in chap_list[ind+1].split(":")]
            ss = (start[0]*3600)+(start[1]*60)+(start[2])
            es = (end[0]*3600)+(end[1]*60)+(end[2])
            if ind != len(chap_list)-1:
                timelines.append([ss, es])
print(timelines)
for i in range(len(timelines)):
    timeline = timelines[i]
    print(timeline)
    cut_clip = video.subclip(timeline[0], timeline[1])
    cut_clip.write_videofile(f"new_clip{i}.mp4")
