"""Fig. 1 of the paper: study design. Counts are typed in from the Phase-0 log."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family":"Arial","font.size":7})
W,H=7.16,3.55
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
INK="#333333"; BLUE="#2b5c8a"; BLUEF="#eaf0f7"; GREEN="#2e6e4e"; GREENF="#eaf4ee"
def box(x,y,w,h,title,lines,fc="white",ec=INK,lw=0.8,ts=7.3,ls=6.8,dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.04",fc=fc,ec=ec,lw=lw,ls="--" if dashed else "-"))
    n=len(lines)+(1 if title else 0); step=0.125; top=y+h/2+(n-1)*step/2
    k=0
    if title:
        ax.text(x+w/2,top,title,ha="center",va="center",fontsize=ts,fontweight="bold",color=INK); k=1
    for i,l in enumerate(lines):
        ax.text(x+w/2,top-(i+k)*step,l,ha="center",va="center",fontsize=ls,color=INK)
def arrow(x0,y0,x1,y1,c=INK):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="-|>",color=c,lw=0.8,shrinkA=0,shrinkB=0,mutation_scale=7))
def line(xs,ys,c=INK): ax.plot(xs,ys,color=c,lw=0.8,solid_capstyle="butt")
# ---- row 1
y1,h1=2.93,0.50; bw=1.27; gap=(W-0.10-5*bw)/4; xs=[0.05+i*(bw+gap) for i in range(5)]
r1=[("Twelve schools",["Bangladesh","5 special, 7 mainstream"]),("Forms received",["n = 863","419 / 444"]),
    ("Analysis cohort",["n = 843, aged 4 to 11","403 / 440"]),("Predictors",["AQ-10 Child, Q1 to Q10","aligned 0 to 3"]),
    ("Label and comparator",["Label: school placement","Rule: AQ-10 score \u2265 6"])]
for x,(t,l) in zip(xs,r1): box(x,y1,bw,h1,t,l)
for i in range(4): arrow(xs[i]+bw,y1+h1/2,xs[i+1],y1+h1/2)
ax.text(xs[0]+0.02,y1-0.09,"Counts are special / mainstream",fontsize=6.2,style="italic",color="#666666",va="center")
ax.text(xs[3]+bw/2,y1-0.09,"SQ1 to SQ4 added for the 14-item set",fontsize=6.2,style="italic",color="#666666",va="center",ha="center")
# exclusion
ex_x=xs[1]+bw+gap/2; line([ex_x,ex_x],[y1+h1/2,y1-0.17])
box(ex_x-0.62,y1-0.47,1.24,0.30,None,["Excluded: age outside 4 to 11","n = 20 (16 / 4)"],fc="#f6f6f6",ec="#888888",dashed=True,ls=6.4)
arrow(ex_x,y1-0.17+0.0,ex_x,y1-0.17)
# split to arms
cx=xs[2]+bw/2; yb=2.28; line([cx,cx],[y1,yb]); 
pA=(0.05,0.98,3.50,1.22); pB=(3.61,0.98,3.50,1.22)
line([pA[0]+pA[2]/2,pB[0]+pB[2]/2],[yb,yb]); arrow(pA[0]+pA[2]/2,yb,pA[0]+pA[2]/2,pA[1]+pA[3]); arrow(pB[0]+pB[2]/2,yb,pB[0]+pB[2]/2,pB[1]+pB[3])
# ---- panel A
ax.add_patch(FancyBboxPatch(pA[:2],pA[2],pA[3],boxstyle="round,pad=0,rounding_size=0.05",fc=BLUEF,ec="#8fa9c6",lw=0.9))
ax.text(pA[0]+0.10,pA[1]+pA[3]-0.12,"A. Internal hold-out validation",fontsize=7.6,fontweight="bold",color=BLUE,va="center")
bx,by,bwid,bh=pA[0]+0.12,pA[1]+0.77,pA[2]-0.24,0.17
ax.add_patch(Rectangle((bx,by),bwid*0.8,bh,fc="#c5d5e8",ec="#8fa9c6",lw=0.6)); ax.add_patch(Rectangle((bx+bwid*0.8,by),bwid*0.2,bh,fc=BLUE,ec="#8fa9c6",lw=0.6))
ax.text(bx+bwid*0.4,by+bh/2,"Development set, n = 674 (322 / 352)",ha="center",va="center",fontsize=6.6,color=INK)
ax.text(bx+bwid*0.9,by+bh/2,"Test, n = 169",ha="center",va="center",fontsize=6.6,color="white")
ax.text(pA[0]+pA[2]/2,by-0.09,"Stratified 80/20 split and five folds fixed with a seed before training",ha="center",va="center",fontsize=6.1,style="italic",color="#555555")
sw=1.02; sg=(pA[2]-0.24-3*sw)/2; sy,sh=pA[1]+0.08,0.48
A3=[("Model selection",["5 learners, 86 settings","5-fold CV by AUROC"]),("Locked model",["RBF SVM, C = 1","fixed threshold 0.50"]),("Single test",["81 / 88 children","evaluated once"])]
ax_=[pA[0]+0.12+i*(sw+sg) for i in range(3)]
for x,(t,l) in zip(ax_,A3): box(x,sy,sw,sh,t,l,ls=6.3,ts=7)
for i in range(2): arrow(ax_[i]+sw,sy+sh/2,ax_[i+1],sy+sh/2)
# ---- panel B
ax.add_patch(FancyBboxPatch(pB[:2],pB[2],pB[3],boxstyle="round,pad=0,rounding_size=0.05",fc=GREENF,ec="#93bba3",lw=0.9))
ax.text(pB[0]+0.10,pB[1]+pB[3]-0.12,"B. Leave-one-school-out validation",fontsize=7.6,fontweight="bold",color=GREEN,va="center")
ax.text(pB[0]+0.10,pB[1]+pB[3]-0.29,"Hyper-parameters locked from arm A",fontsize=6.1,style="italic",color="#555555",va="center")
g0x,g0y,cs=pB[0]+0.30,pB[1]+0.07,0.051
for i in range(12):
    for j in range(12):
        ax.add_patch(Rectangle((g0x+j*cs,g0y+(11-i)*cs),cs*0.92,cs*0.92,fc=GREEN if i==j else "#d3e6da",ec="none"))
ax.text(g0x+6*cs,g0y+12*cs+0.06,"Schools S01 to S12",ha="center",va="center",fontsize=6.1,color="#555555")
ax.text(g0x-0.09,g0y+6*cs,"Folds 1 to 12",rotation=90,ha="center",va="center",fontsize=6.1,color="#555555")
lx=g0x+12*cs+0.12
ax.add_patch(Rectangle((lx,g0y+0.36),0.08,0.08,fc=GREEN,ec="none")); ax.text(lx+0.12,g0y+0.40,"held-out school",fontsize=6.3,va="center")
ax.add_patch(Rectangle((lx,g0y+0.18),0.08,0.08,fc="#d3e6da",ec="none")); ax.text(lx+0.12,g0y+0.22,"training schools",fontsize=6.3,va="center")
rb_x=pB[0]+pB[2]-1.30
box(rb_x,pB[1]+0.46,1.20,0.44,"Refit per fold",["on 11 schools","727 to 806 children"],ls=6.3,ts=7)
box(rb_x,pB[1]+0.05,1.20,0.34,"Pooled predictions",["out-of-fold, n = 843"],ls=6.3,ts=7)
arrow(rb_x+0.60,pB[1]+0.46,rb_x+0.60,pB[1]+0.39)
# ---- row 3 analyses
y3,h3=0.04,0.52; aw=(W-0.10-3*0.09)/4; axs=[0.05+i*(aw+0.09) for i in range(4)]
A=[("Model versus AQ-10 rule",["McNemar within each group","DeLong test, bootstrap CIs"]),
   ("Lower cut-offs (post hoc)",["score \u2265 5 at matched specificity","score \u2265 4 by Youden index"]),
   ("Supplementary items",["10 versus 14 items, same children","pre-specified adoption rule"]),
   ("Explanation and audit",["Kernel SHAP, permutation importance","per-school errors, subgroups"])]
for x,(t,l) in zip(axs,A): box(x,y3,aw,h3,t,l,ls=6.4,ts=7)
# connectors: A -> analysis 1 ; B -> all four
yA=0.80; line([pA[0]+pA[2]/2,pA[0]+pA[2]/2],[pA[1],yA+0.07],BLUE); line([axs[0]+aw*0.35,pA[0]+pA[2]/2],[yA+0.07,yA+0.07],BLUE); arrow(axs[0]+aw*0.35,yA+0.07,axs[0]+aw*0.35,y3+h3,BLUE)
bcx=pB[0]+pB[2]/2; line([bcx,bcx],[pB[1],yA]); line([axs[0]+aw*0.65,axs[3]+aw/2],[yA,yA],GREEN); line([bcx,bcx],[pB[1],yA],GREEN)
for k,x in enumerate(axs):
    xx=x+aw*0.65 if k==0 else x+aw/2
    arrow(xx,yA,xx,y3+h3,GREEN)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "fig1_study_design.png", dpi=600)
print("ok")
