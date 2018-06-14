import requests,json,wave,io,os,base64,urllib.request

from pydub import AudioSegment


def mp3_2_wav(audio,new_file):
    '''
    mp3转wav格式
    '''
    fp=open(audio,'rb')
    data=fp.read()
    fp.close()

    aud=io.BytesIO(data)
    sound=AudioSegment.from_file(aud,format='mp3')
    raw_data = sound._data

    l=len(raw_data)
    f=wave.open(new_file,'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(8000)
    f.setnframes(l)
    f.writeframes(raw_data)
    f.close()

def get_response(msg):
    '''
    图灵机器人API调用
    '''
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key'  : '36dbe7e4ab2e4449861d05ebd14500e1',
        'info'  : msg,
        'userid' : 'resnick',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        return r.get('text')
    except:
        return

def record_api():
    '''
    调用百度语音识别API
    '''
    #获得TOKEN
    apiKey='seIZ0TyG2V1tIEA3OtHw6n4m'
    secretKey='cbfb0d2544c7a49db9b1700d5c938ab4'
    url="https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + apiKey + "&client_secret=" + secretKey
    res=urllib.request.urlopen(url).read()
    data=json.loads(res.decode("utf-8"))
    token=data['access_token']

    #调用API 识别语音
    voice_rate=8000
    wave_file='./audio/audio_convert.wav'
    user_id='resnick'
    wave_type='wav'

    f=open(wave_file,'rb')
    speech=base64.b64encode(f.read()).decode('utf-8')
    size=os.path.getsize(wave_file)
    update=json.dumps({
        'format':wave_type,
        'rate':voice_rate,
        'channel':1,
        'cuid':user_id,
        'token':token,
        'speech':speech,
        'len':size,
        'lan':'zh'
    }).encode('utf-8')
    svl_url="http://vop.baidu.com/server_api"
    r=urllib.request.urlopen(svl_url,update)
    #解析获得的json
    t=r.read().decode('utf-8')
    result=json.loads(t)
    print(result)
    if result['err_msg']=='success.':
        word=result['result'][0].encode('utf-8')
        if word!='':
            word=word.decode('utf-8')
            if word[len(word)-1:len(word)]=='，':
                print(word[0:len(word)-1])
                return word[0:len(word)-1]
            else:
                print(word)
                return word
        else:
            print('转换错误')
            return ''
    else:
        print('ERROR')
