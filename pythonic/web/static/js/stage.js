// The panel is drawn at a fixed design size and scaled to the window,
// letterboxed: the whole stage stays visible and centred at one scale.

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

/** Keep a stage element fitted to the window; returns a function that stops it. */
export function mountStage(stage, win = window) {
  stage.style.width = `${STAGE_WIDTH}px`;
  stage.style.height = `${STAGE_HEIGHT}px`;
  const fit = () => {
    const { scale, x, y } = fitStage(win.innerWidth, win.innerHeight);
    stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
    stage.dataset.scale = String(scale);
  };
  fit();
  win.addEventListener('resize', fit);
  return () => win.removeEventListener('resize', fit);
}
