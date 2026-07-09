"""
B站视频分析流水线 — 共享脚本
处理：下载 → 拆帧 → 视觉识别 → 音频转录
输出原始数据供 AI 生成报告
"""
import urllib.request, json, os, sys, subprocess, time, re
from pathlib import Path

def get_video_info(bvid: str, cookies_path: str = None) -> dict:
    """获取B站视频元数据"""
    url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': 'https://www.bilibili.com/'
    })
    data = json.loads(urllib.request.urlopen(req).read())['data']
    return {
        'bvid': bvid,
        'title': data['title'],
        'owner': data['owner']['name'],
        'duration': data['duration'],
        'cid': data['cid'],
        'aid': data['aid'],
        'pic': data['pic'],
        'desc': data.get('desc', ''),
        'has_subtitle': len(data.get('subtitle', {}).get('list', [])) > 0
    }

def download_video(bvid: str, output_dir: str, cookies_path: str = None) -> dict:
    """使用 you-get 下载视频和音频，回退到 yt-dlp"""
    os.makedirs(output_dir, exist_ok=True)

    url = f'https://www.bilibili.com/video/{bvid}'
    video_path = os.path.join(output_dir, 'video.mp4')
    audio_path = os.path.join(output_dir, 'audio.m4a')

    # 尝试 you-get
    try:
        cmd = [sys.executable, '-m', 'you_get', '-o', output_dir, url]
        if cookies_path:
            cmd.extend(['--cookies', cookies_path])
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except:
        pass

    # 回退：使用 B站 API + urllib 直接下载
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1024*1024:
        _download_via_api(bvid, get_video_info(bvid)['cid'], output_dir, cookies_path)

    # 查找下载的文件
    result = {'video': None, 'audio': None}
    for f in os.listdir(output_dir):
        full = os.path.join(output_dir, f)
        if f.endswith('.mp4') and 'audio' not in f.lower():
            result['video'] = full
        elif f.endswith(('.m4a', '.mp3', '.aac')):
            result['audio'] = full

    return result

def _download_via_api(bvid: str, cid: int, output_dir: str, cookies_path: str):
    """通过 B站 API 直接下载"""
    # 获取视频流 URL
    url = f'https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn=80&platform=html5'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': f'https://www.bilibili.com/video/{bvid}'
    })
    data = json.loads(urllib.request.urlopen(req).read())
    video_url = data['data']['durl'][0]['url']

    video_path = os.path.join(output_dir, 'video.mp4')
    req2 = urllib.request.Request(video_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer': f'https://www.bilibili.com/video/{bvid}'
    })
    with open(video_path, 'wb') as f:
        f.write(urllib.request.urlopen(req2).read())

def extract_frames(video_path: str, output_dir: str, interval_sec: int = 18,
                   similarity_threshold: float = 0.85) -> list:
    """提取关键帧，带去重"""
    import cv2, numpy as np

    os.makedirs(output_dir, exist_ok=True)
    # 清空旧帧
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)

    frames = []
    count = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keep = True
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                similarity = 1.0 - (np.count_nonzero(diff > 25) / diff.size)
                keep = similarity < similarity_threshold

            if keep:
                ts = count / fps
                fname = f'{len(frames)+1:03d}_{int(ts//60)}m{int(ts%60):02d}s.jpg'
                out_path = os.path.join(output_dir, fname)
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append({'path': out_path, 'time': ts, 'filename': fname})
                prev_gray = gray

        count += 1

    cap.release()
    return frames

def transcribe_audio(audio_path: str, output_dir: str) -> str:
    """Whisper 语音转文字"""
    import whisper

    model = whisper.load_model('tiny')
    result = model.transcribe(audio_path, language='zh')

    # 保存纯文本
    txt_path = os.path.join(output_dir, 'transcript.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    # 保存带时间戳版本
    ts_path = os.path.join(output_dir, 'transcript_timed.txt')
    with open(ts_path, 'w', encoding='utf-8') as f:
        for seg in result['segments']:
            m, s = divmod(int(seg['start']), 60)
            f.write(f'[{m:02d}:{s:02d}] {seg["text"].strip()}\n')

    return result['text']

def vision_analyze(image_path: str, prompt: str, provider: str = 'glm') -> str:
    """调用视觉模型分析图片（GLM 优先）"""
    sys.path.insert(0, str(Path(__file__).parent))
    from vision_provider import vision_analyze as _vision
    return _vision(image_path, prompt, provider)

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='B站视频分析流水线')
    parser.add_argument('bvid', help='B站视频 BV号')
    parser.add_argument('-o', '--output', default='./output', help='输出目录')
    parser.add_argument('-c', '--cookies', help='cookies.txt 路径')
    parser.add_argument('--interval', type=int, default=18, help='拆帧间隔(秒)')
    parser.add_argument('--vision-provider', default='glm', choices=['glm', 'qwen', 'openai', 'doubao'])
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-transcribe', action='store_true')

    args = parser.parse_args()
    bvid = args.bvid
    out = args.output

    print(f'=== B站视频分析: {bvid} ===')

    # 1. 获取信息
    info = get_video_info(bvid, args.cookies)
    print(f'标题: {info["title"]}')
    print(f'UP主: {info["owner"]}')
    print(f'时长: {info["duration"]}s')

    # 2. 下载
    if not args.skip_download:
        files = download_video(bvid, out, args.cookies)
        print(f'视频: {files["video"]}')
        print(f'音频: {files["audio"]}')

    # 3. 拆帧
    video = os.path.join(out, 'video.mp4')
    frames_dir = os.path.join(out, 'frames')
    if os.path.exists(video):
        frames = extract_frames(video, frames_dir, args.interval)
        print(f'关键帧: {len(frames)} 帧')

    # 4. 转录
    audio = os.path.join(out, 'audio.m4a')
    if not args.skip_transcribe and os.path.exists(audio):
        text = transcribe_audio(audio, out)
        print(f'转录: {len(text)} 字')

    print('=== 流水线完成 ===')
    print(f'输出目录: {out}')
