import os
import tempfile
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse,HTMLResponse
from funasr import AutoModel
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()

#app.mount("/static", StaticFiles(directory="./static"), name="static")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="ct-punc",
    device=device,
    # spk_model="cam++",
)


def convert_audio(input_file):
    import ffmpeg

    output_file = input_file + ".wav"
    (
        ffmpeg.input(input_file)
        .output(output_file)
        .run(quiet=True)
    )
    return output_file

# 异步函数，用于保存上传的文件到临时目录
async def save_upload_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[1]  # 获取文件后缀名
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:  # 创建临时文件
        temp_file.write(await upload_file.read())  # 将上传的文件内容写入临时文件
        return temp_file.name  # 返回临时文件路径



def format_timestamp(milliseconds):
    # 将毫秒转换为SRT格式的时间戳
    seconds, milliseconds = divmod(milliseconds, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def funasr_to_srt(data):
    text = data[0]['text']
    timestamps = data[0]['timestamp']
    punctuations = ".,。，!！?？、"
    sentences = []
    start_idx = 0
    for i, char in enumerate(text):
        if char in punctuations:
            sentences.append(text[start_idx:i])
            start_idx = i+1
    if start_idx < len(text):
        sentences.append(text[start_idx:])

    srt_data = []
    all_len=0
    end=0
    for idx, sentence in enumerate(sentences):
        try:
            if len(sentence.strip())==0:
                continue
            start=end
            end+=len(sentence)
            first_char_time = timestamps[start][0]
            if end-1> len(timestamps):
                last_char_time = timestamps[-1][1]
            else:
                last_char_time=timestamps[end-1][1]
            
        except IndexError:
            continue

        start_time = format_timestamp(first_char_time)
        end_time = format_timestamp(last_char_time)

        srt_data.append(f"{idx+1}\n{start_time} --> {end_time}\n{sentence.strip()}\n\n")

    return ''.join(srt_data)



@app.get("/", response_class=HTMLResponse)
async def root():
    # upload form template
    return """
    <!doctype html>
    <title>Upload Audio File</title>
    <h1>Upload Audio File</h1>
    <script
  src="https://code.jquery.com/jquery-3.7.1.js"
  integrity="sha256-eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4="
  crossorigin="anonymous"></script>
    <form id="fileUploadForm" enctype="multipart/form-data">
        <input type="file" name="file" id="file">
        <input value="Upload" id="uploadBtn" name="uploadBtn" type="button" />
        <span id="message"></span>
        <input value="Download" id="downloadBtn" name="downloadBtn" type="button" />

        <div style="display:block;width: 100%;height: 500px;">
            <textarea name="text" id="text" cols="30" rows="10" style="height: 100%;width: 49.5%;float:left"></textarea>
            <textarea name="srt" id="srt" cols="30" rows="10" style="height: 100%;width: 49.5%;float:right" ></textarea>
        </div>
    </form>

    <script>

    $(document).ready(function() {
        function changeFileExtension(filePath) {
            // 使用正则表达式分割路径和文件名
            var parts = filePath.split('\\\\');
            if (parts.length === 1) {
                parts = filePath.split("/");
            }
            const fileNameWithExtension = parts[parts.length - 1]; // 获取最后一部分作为文件名
            
            // 分割文件名和扩展名
            const fileNameParts = fileNameWithExtension.split('.');
            
            // 如果文件名没有扩展名，则直接添加 .srt
            if (fileNameParts.length === 1) {
                return `${fileNameWithExtension}.srt`;
            }
            
            // 构建新的文件名，将扩展名替换为 .srt
            const newFileName = `${fileNameParts.slice(0, -1).join('.')}.srt`;
            
            return newFileName;
        }
        function saveFile(text, filename) {
            var blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
            var downloadLink = $('<a></a>')
                .attr('href', URL.createObjectURL(blob))
                .attr('download', filename)
                .appendTo('body');
            downloadLink[0].click();
            downloadLink.remove();
        }
        $('#uploadBtn').click(function() {
            
            var formData = new FormData($('#fileUploadForm')[0]);

            $('#message').html('Processing...');
    
            $.ajax({
                url: '/asr', 
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    $('#message').html('Success');
                    var result=response['result'][0];
                    $('#text').html(result['text']);
                    $('#srt').html(result['srt']);
                },
                error: function(xhr, status, error) {
                $('#message').html('Processing Error');
                    console.error('File upload failed:', status, error);
                }
            });
        });

        $('#downloadBtn').click(function() {
            var filename=$('#file').val();
            saveFile($('#srt').val(), changeFileExtension(filename));
        });
    });
    </script>

    """
    

@app.post("/asr")
async def asr(file: List[UploadFile] = File(...)):
    temp_input_file_path = None
    try:
        if not file or any(f.filename == "" for f in file):
            raise Exception("No file was uploaded")
        if len(file) != 1:
            raise Exception("Only one file can be uploaded at a time")
        file = file[0]

        ext_name = os.path.splitext(file.filename)[1].strip('.')

        temp_input_file_path = await save_upload_file(file)  # 保存上传的文件
        if ext_name not in ['wav', 'mp3']:
            # 如果不是音频文件,用ffmpeg转换为音频文件
            temp_input_file_path = convert_audio(temp_input_file_path)
            # raise Exception("Unsupported file extension")

        print(temp_input_file_path)

        result = model.generate(
            input=temp_input_file_path,
            batch_size_s=300,
            batch_size_threshold_s=60,
            # hotword='魔搭'
        )

        try:
            srt=funasr_to_srt(result)
            result[0]['srt']=srt
        except:
            print('srt convert fail')

        return {"result": result}  # 返回识别结果
    except Exception as e:  # 捕获其他异常
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        for temp_file in [temp_input_file_path]:
            if temp_file and os.path.exists(temp_file):  # 检查路径是否存在
                os.remove(temp_file)  # 删除文件


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12369)  # 运行FastAPI应用
