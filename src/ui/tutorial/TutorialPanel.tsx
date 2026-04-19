// Tutorial panel — guided task list for new users.
// Tracks progress in localStorage. Auto-detects some tasks via events.
// Can be dismissed and reopened from toolbar.
// Draggable by header.

import { useState, useEffect, useCallback, useRef } from 'react';
import { ChevronDown, ChevronRight, X, RotateCcw, Check, Circle, Trophy } from 'lucide-react';
import { TUTORIAL_GOALS, ALL_TASK_IDS, type TutorialGoal } from './tutorialData';
import './TutorialPanel.css';

const STORAGE_KEY = 'nodetorch-tutorial';
const DISMISSED_KEY = 'nodetorch-tutorial-dismissed';
const SEEN_KEY = 'nodetorch-tutorial-seen';  // set after first render

interface TutorialState {
  completed: Set<string>;
}

function loadState(): TutorialState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return { completed: new Set(parsed.completed ?? []) };
    }
  } catch { /* ignore */ }
  return { completed: new Set() };
}

function saveState(state: TutorialState) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ completed: Array.from(state.completed) }));
}

function isDismissed(): boolean {
  return localStorage.getItem(DISMISSED_KEY) === 'true';
}

/** True on the very first visit (no localStorage keys yet). */
function isFirstVisit(): boolean {
  localStorage.removeItem('nodetorch-tutorial-seen')
  localStorage.removeItem('nodetorch-tutorial')
  localStorage.removeItem('nodetorch-tutorial-dismissed')

  return !localStorage.getItem(SEEN_KEY);
}

export function TutorialPanel() {
  const [state, setState] = useState<TutorialState>(loadState);
  const firstVisit = useRef(isFirstVisit());
  const [dismissed, setDismissed] = useState(() => firstVisit.current ? false : isDismissed());
  const [collapsed, setCollapsed] = useState(() => !firstVisit.current);
  const [expandedGoal, setExpandedGoal] = useState<string | null>(() => {
    const s = loadState();
    for (const goal of TUTORIAL_GOALS) {
      if (goal.tasks.some(t => !s.completed.has(t.id))) return goal.id;
    }
    return null;
  });

  // Mark as seen after first render
  useEffect(() => {
    localStorage.setItem(SEEN_KEY, 'true');
  }, []);

  // Persist state changes
  useEffect(() => { saveState(state); }, [state]);

  // --- Dragging ---
  const panelRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const dragState = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);
  const didDrag = useRef(false);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.tutorial-icon-btn')) return;
    e.preventDefault();
    const panel = panelRef.current;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    dragState.current = {
      startX: e.clientX,
      startY: e.clientY,
      origX: rect.left,
      origY: rect.top,
    };
    didDrag.current = false;

    const onMove = (ev: MouseEvent) => {
      if (!dragState.current) return;
      const dx = ev.clientX - dragState.current.startX;
      const dy = ev.clientY - dragState.current.startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) didDrag.current = true;
      setPos({ x: dragState.current.origX + dx, y: dragState.current.origY + dy });
    };
    const onUp = () => {
      dragState.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  const onHeaderClick = useCallback(() => {
    if (didDrag.current) return; // suppress click after drag
    setCollapsed(c => !c);
  }, []);

  // Toggle a task
  const toggleTask = useCallback((taskId: string) => {
    setState(prev => {
      const next = new Set(prev.completed);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return { completed: next };
    });
  }, []);

  // Auto-detect events
  const completeTask = useCallback((taskId: string) => {
    setState(prev => {
      if (prev.completed.has(taskId)) return prev;
      const next = new Set(prev.completed);
      next.add(taskId);
      return { completed: next };
    });
  }, []);

  // Listen for auto-detect events on window
  useEffect(() => {
    const handler = (e: Event) => {
      const eventKey = (e as CustomEvent).detail;
      for (const goal of TUTORIAL_GOALS) {
        for (const task of goal.tasks) {
          if (task.autoDetect === eventKey) {
            completeTask(task.id);
          }
        }
      }
    };
    window.addEventListener('nodetorch-tutorial', handler);
    return () => window.removeEventListener('nodetorch-tutorial', handler);
  }, [completeTask]);

  const dismiss = () => {
    localStorage.setItem(DISMISSED_KEY, 'true');
    setDismissed(true);
  };

  const reset = () => {
    setState({ completed: new Set() });
    localStorage.removeItem(DISMISSED_KEY);
    setDismissed(false);
    setCollapsed(false);
    setExpandedGoal(TUTORIAL_GOALS[0].id);
  };

  // Reopen from toolbar button
  useEffect(() => {
    const handler = () => {
      setDismissed(false);
      localStorage.removeItem(DISMISSED_KEY);
      setCollapsed(false);
    };
    window.addEventListener('nodetorch-tutorial-reopen', handler);
    return () => window.removeEventListener('nodetorch-tutorial-reopen', handler);
  }, []);

  if (dismissed) return null;

  const totalTasks = ALL_TASK_IDS.length;
  const completedCount = ALL_TASK_IDS.filter(id => state.completed.has(id)).length;
  const allDone = completedCount === totalTasks;
  const progressPct = Math.round((completedCount / totalTasks) * 100);

  const posStyle = pos
    ? { top: pos.y, left: pos.x, transform: 'none' }
    : undefined;

  return (
    <div
      ref={panelRef}
      className={`tutorial-panel ${collapsed ? 'tutorial-panel-collapsed' : ''}`}
      style={posStyle}
    >
      <div className="tutorial-header" onMouseDown={onDragStart} onClick={onHeaderClick}>
        <div className="tutorial-header-left">
          {allDone ? <Trophy size={14} className="tutorial-trophy" /> : <span className="tutorial-progress-ring">{progressPct}%</span>}
          <span className="tutorial-header-title">
            {allDone ? 'Tutorial Complete!' : 'Getting Started'}
          </span>
        </div>
        <div className="tutorial-header-right">
          <button className="tutorial-icon-btn" onClick={(e) => { e.stopPropagation(); reset(); }} title="Reset tutorial">
            <RotateCcw size={12} />
          </button>
          <button className="tutorial-icon-btn" onClick={(e) => { e.stopPropagation(); dismiss(); }} title="Dismiss tutorial">
            <X size={12} />
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="tutorial-body">
          <div className="tutorial-progress-bar-track">
            <div className="tutorial-progress-bar-fill" style={{ width: `${progressPct}%` }} />
          </div>

          {TUTORIAL_GOALS.map(goal => (
            <GoalSection
              key={goal.id}
              goal={goal}
              completed={state.completed}
              expanded={expandedGoal === goal.id}
              onToggleExpand={() => setExpandedGoal(expandedGoal === goal.id ? null : goal.id)}
              onToggleTask={toggleTask}
            />
          ))}

          {allDone && (
            <div className="tutorial-congrats">
              You've completed all the basics! Explore presets, try different architectures, and experiment freely.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function GoalSection({ goal, completed, expanded, onToggleExpand, onToggleTask }: {
  goal: TutorialGoal;
  completed: Set<string>;
  expanded: boolean;
  onToggleExpand: () => void;
  onToggleTask: (id: string) => void;
}) {
  const goalDone = goal.tasks.every(t => completed.has(t.id));
  const goalProgress = goal.tasks.filter(t => completed.has(t.id)).length;

  return (
    <div className={`tutorial-goal ${goalDone ? 'tutorial-goal-done' : ''}`}>
      <button className="tutorial-goal-header" onClick={onToggleExpand}>
        <span className="tutorial-goal-chevron">
          {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </span>
        <span className="tutorial-goal-title">{goal.title}</span>
        <span className="tutorial-goal-count">{goalProgress}/{goal.tasks.length}</span>
      </button>

      {expanded && (
        <div className="tutorial-tasks">
          <div className="tutorial-goal-desc">{goal.description}</div>
          {goal.tasks.map(task => {
            const done = completed.has(task.id);
            return (
              <div key={task.id} className={`tutorial-task ${done ? 'tutorial-task-done' : ''}`}>
                <button className="tutorial-task-toggle" onClick={() => onToggleTask(task.id)}>
                  <span className="tutorial-task-check">
                    {done ? <Check size={12} /> : <Circle size={12} />}
                  </span>
                  <span className="tutorial-task-text">{task.text}</span>
                </button>
                {task.hint && !done && <div className="tutorial-task-hint">{task.hint}</div>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Dispatch a tutorial auto-detect event. Call this from other components. */
export function tutorialEvent(key: string) {
  window.dispatchEvent(new CustomEvent('nodetorch-tutorial', { detail: key }));
}
