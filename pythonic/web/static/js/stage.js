// The panel is drawn at a fixed design size and scaled to the window,
// letterboxed: the whole stage stays visible and centred at one scale. The
// height is 1000 with the edit rack drawer open and 700 (the face) with it
// closed (setStageHeight).

export const STAGE_WIDTH = 1600;
export const STAGE_HEIGHT = 1000;

/** Scale and offset that fit the stage into a view, centred. */
export function fitStage(viewWidth, viewHeight, width = STAGE_WIDTH, height = STAGE_HEIGHT) {
  const scale = Math.max(0, Math.min(viewWidth / width, viewHeight / height));
  return {
    scale,
    x: (viewWidth - width * scale) / 2,
    y: (viewHeight - height * scale) / 2,
  };
}

const fitters = new WeakMap();

/** The stage's design height now. */
export const stageHeight = (stage) => Number(stage.dataset.height) || STAGE_HEIGHT;

/** Change the stage's design height and fit it again. */
export function setStageHeight(stage, height) {
  stage.dataset.height = String(height);
  stage.style.height = `${height}px`;
  const fit = fitters.get(stage);
  if (fit) fit();
}

/** Keep a stage element fitted to the window; returns a function that stops it. */
export function mountStage(stage, win = window) {
  stage.style.width = `${STAGE_WIDTH}px`;
  stage.style.height = `${stageHeight(stage)}px`;
  const fit = () => {
    const { scale, x, y } = fitStage(win.innerWidth, win.innerHeight, STAGE_WIDTH, stageHeight(stage));
    stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    stage.dataset.scale = String(scale);
  };
  fit();
  fitters.set(stage, fit);
  win.addEventListener('resize', fit);
  return () => { fitters.delete(stage); win.removeEventListener('resize', fit); };
}
