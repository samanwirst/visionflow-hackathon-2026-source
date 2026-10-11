"""QA data, not predictions: original procedural service-bay sessions.

Human review is required; proposed labels are never marked verified here.
Synthetic occupancy is a narrow AutoLearn demonstration, not natural CV accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True, help='New owned output directory; includes proposed, unverified labels.')
parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
args = parser.parse_args()
OUT=args.out.resolve()
if OUT.exists() and any(OUT.iterdir()):
 raise SystemExit('Output directory must be empty; existing evidence is never overwritten.')
OUT.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype(str(args.font),14)
samples=[]
sheet=Image.new('RGB',(4*352,6*245),(15,20,29))
for session in range(12):
 rng=random.Random(4102026+session)
 wall=tuple(rng.randint(60,140) for _ in range(3))
 floor=tuple(rng.randint(35,110) for _ in range(3))
 pole_x=[rng.randrange(4,100),rng.randrange(235,315)]
 car_width=rng.randint(155,210);car_height=rng.randint(38,55)
 car_color=rng.choice([(195,60,45),(40,110,190),(50,150,95),(190,150,45),(180,180,190)])
 for frame in range(2):
  present=frame==0
  im=Image.new('RGB',(320,192),wall);d=ImageDraw.Draw(im)
  d.rectangle((0,100,320,192),fill=floor)
  # Session-specific workshop geometry, no faces, plates, names or reused assets.
  for k in range(8):
   x=rng.randrange(2,290);y=rng.randrange(4,87)
   d.rectangle((x,y,x+rng.randint(6,25),y+rng.randint(5,19)),fill=tuple(rng.randint(20,180) for _ in range(3)))
  for x in pole_x:d.rectangle((x,8,x+5,107),fill=(180,185,190))
  d.polygon([(35,185),(65,112),(266,112),(302,185)],outline=(230,210,130),width=3)
  for k in range(3):
   x=rng.randrange(30,280)
   d.line((x,186,x+rng.randint(-30,30),113),fill=tuple(min(255,c+12) for c in floor),width=1)
  if present:
   left=(320-car_width)//2+rng.randint(-7,7);right=left+car_width
   bottom=157;top=bottom-car_height
   d.ellipse((left+12,bottom-3,right-7,bottom+15),fill=tuple(max(0,c-15) for c in floor))
   d.rounded_rectangle((left,top+12,right,bottom),radius=10,fill=car_color)
   d.polygon([(left+27,top+16),(left+55,top-20),(right-46,top-20),(right-22,top+16)],fill=car_color)
   d.polygon([(left+41,top+11),(left+61,top-13),(right-56,top-13),(right-36,top+11)],fill=(40,56,72))
   d.line((left+car_width//2,top-13,left+car_width//2,top+11),fill=car_color,width=4)
   for x in [left+37,right-38]:
    d.ellipse((x-15,bottom-13,x+15,bottom+17),fill=(15,17,20))
    d.ellipse((x-7,bottom-5,x+7,bottom+9),fill=(140,150,158))
   d.rectangle((right-11,top+23,right-2,top+31),fill=(232,230,191))
   d.rectangle((left+2,top+23,left+7,top+31),fill=(175,37,30))
   d.line((left+67,top+17,left+67,bottom-8),fill=tuple(max(0,c-25) for c in car_color),width=1)
  sample_id=f's{session:02d}-f{frame:02d}'
  path=OUT/(sample_id+'.png');im.save(path)
  samples.append({'sample_id':sample_id,'path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
   'source_id':f'procedural-session-{session:02d}','session_id':f'procedural-session-{session:02d}',
   'source_asset_id':f'original-render-seed-{4102026+session}','video_timestamp':float(frame),
   'split':'train' if session<8 else ('validation' if session<10 else 'test'),
   'proposed_label':'occupied' if present else 'empty','human_verified':False,
   'provenance':'Original procedural image; synthetic service bay; no third-party image assets; NOT natural video.'})
  cell=session*2+frame;x=(cell%4)*352;y=(cell//4)*245
  sheet.paste(im,(x+16,y+12));sd=ImageDraw.Draw(sheet)
  sd.text((x+16,y+211),f'{sample_id}  |  proposed: '+('occupied' if present else 'empty'),font=font,fill=(229,234,242))
manifest={'dataset_id':'synthetic-occupancy-v1','task':'Is a vehicle present inside the marked service bay?',
 'labels':['occupied','empty'],'samples':samples,'verified':False,
 'limits':'24 procedural examples, 12 independent rendering sessions. Measures only this generator domain; no natural-video robustness or savings claims.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');sheet.save(OUT/'review-sheet.png')
print('24 procedural samples prepared; none verified. Manifest and review sheet:',OUT)
