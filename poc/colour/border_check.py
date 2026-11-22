import sys, numpy as np, cv2
from PIL import Image
sys.path.insert(0, sys.argv[1])  # usage: border_check.py <poc/colour dir> <results dir>
from common import signed_distance
RES = sys.argv[2]
def rough_from(prefix, kind):
    O = cv2.imread(f"{RES}/{prefix}_orig_editframe.png"); H, W = O.shape[:2]
    if kind.endswith(".png"):
        ell = np.asarray(Image.open(f"{RES}/../{kind}"))[..., 3] == 0
        return cv2.resize(ell.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    soft = np.load(f"{RES}/{prefix}_soft.npy")
    return None
for prefix, kind in [("fin_balloon", "demo_mona_lisa_balloon.png"), ("fin_mona", "demo_mona_lisa_shoulder.png")]:
    m = rough_from(prefix, kind); H, W = m.shape
    # outline pixels: inside pixels with an outside 4-neighbour, plus inside pixels on the frame border
    er = cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    real = (m > 0) & (er == 0)
    border = np.zeros_like(m, bool); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    on_border = (m > 0) & border
    d = signed_distance(m)
    band = (d > 0) & (d < 80)
    # what a full ring would hold if the frame went on: pad the frame and redo
    P = 400
    mp = cv2.copyMakeBorder(m, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    # the mask continues out of frame only for ellipses; rebuild by edge replication of the mask
    mp = cv2.copyMakeBorder(m, P, P, P, P, cv2.BORDER_REPLICATE)
    dp = signed_distance(mp)
    full_band = ((dp > 0) & (dp < 80)).sum()
    print(f"{prefix}: mask on the frame border {on_border.sum()} px (outline in the frame {real.sum()} px); "
          f"band 0-80 px holds {band.sum()} px vs {full_band} if the frame went on ({band.sum()/full_band*100:.0f}%)")
    if on_border.any():
        ys, xs = np.nonzero(on_border)
        i = len(ys) // 2
        print(f"   distance inside at a border pixel ({xs[i]},{ys[i]}): {-d[ys[i], xs[i]]:.0f} px to the nearest real mask edge "
              f"(the frame border is not counted as an edge)")
