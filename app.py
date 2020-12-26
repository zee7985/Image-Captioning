from flask import Flask,render_template,request 

import Caption
from gtts import gTTS
app =Flask(__name__)  


@app.route('/') 
def hello():
	return render_template("index1.html")


@app.route('/',methods=['POST']) 
def marks():
	if request.method=='POST':
		f=request.files['image']
		# path=r"C:\Users\zeesh\Machine Learning\Image Captioning\InceptionV3\static\{}".format(f.filename) 
		path="./static/{}".format(f.filename) 

		f.save(path)

		# path2="./static/{}"+".mp3"

		caption=Caption.captionIt(path)
		# outputAudio=gTTS(text=caption,lang='en',slow=False)

		# outputAudio.save(path2)


		# print(caption)
		result_dic ={
			'image' : path,
			'caption' : caption,
			# 'sound': path2
		}
		
	return render_template("index1.html",your_result=result_dic)

if __name__=='__main__':
	app.debug=True     
	app.run()