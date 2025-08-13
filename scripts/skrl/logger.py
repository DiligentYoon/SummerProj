# --- 필요 모듈 ---
import os, copy
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Excel 로거 ---
class PerEpisodeExcelLogger:
    """
    (env_id, episode_id)별로 스텝 행들을 쌓아두다가, 에피소드 종료 시
    metrics.xlsx 에 sheet=env{env_id}_ep{episode_id} 로 기록.
    별도의 SUMMARY 시트(에피소드 통계)도 저장.
    """
    def __init__(self, path="metrics.xlsx"):
        self.path = path
        self.buffers = defaultdict(list)   # key=(env_id, ep_id) -> list of dict rows
        self.summary_rows = []            # 에피소드별 요약행 저장

    def log_step(self, env_id, ep_id, step_idx, row_dict):
        row = {"env_id": env_id, "episode_id": ep_id, "step": step_idx}
        row.update(row_dict)
        self.buffers[(env_id, ep_id)].append(row)

    def end_episode(self, env_id, ep_id):
        key = (env_id, ep_id)
        if key not in self.buffers or len(self.buffers[key]) == 0:
            return
        df = pd.DataFrame(self.buffers.pop(key))

        # 시트명(엑셀 31자 제한 고려 시 필요시 축약)
        sheet_name = f"env{env_id}_ep{ep_id}"
        if len(sheet_name) > 31:
            sheet_name = f"e{env_id}_p{ep_id}"

        # 파일 존재 여부에 따라 write/append
        mode = "a" if os.path.exists(self.path) else "w"
        with pd.ExcelWriter(self.path, engine="openpyxl", mode=mode, if_sheet_exists=("replace" if mode=="a" else None)) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            # SUMMARY는 매번 덮어쓰기
            if len(self.summary_rows) > 0:
                pd.DataFrame(self.summary_rows).to_excel(writer, sheet_name="SUMMARY", index=False)

        # 간단 요약행(원하시면 지표 추가/수정)
        ep_len = len(df)
        total_reward = df["reward"].sum() if "reward" in df.columns else np.nan
        self.summary_rows.append({
            "env_id": env_id,
            "episode_id": ep_id,
            "length": ep_len,
            "total_reward": total_reward,
            "last_phase": df["phase_tp1"].iloc[-1] if "phase_tp1" in df.columns else np.nan
        })

    def close(self):
        # 남은 버퍼가 있으면 모두 기록
        for (env_id, ep_id) in list(self.buffers.keys()):
            self.end_episode(env_id, ep_id)
