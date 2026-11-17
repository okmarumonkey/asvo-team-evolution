from collections.abc import Sequence
import re
import functools

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import concurrency, helper_functions
from typing_extensions import override

DEFAULT_PRE_ACT_KEY = 'Act'
from collections.abc import Mapping
from concordia.components import agent as agent_components
from examples.ASVO.value_components import desire_svo_comp

SVO_RANGES_QUARTILE = {
        "Altruistic": (73.09, 84.32),
        "Prosocial": (50.62, 61.85),
        "Individualistic": (28.15, 39.38),
        "Competitive": (5.68, 16.91),
}

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

class MCTSActComponent(entity_component.ActingComponent):
    def __init__(
            self,
            model: language_model.LanguageModel,
            clock: game_clock.GameClock,
            num_proposed_actions: int,
            desire_component_dict: Mapping[str, entity_component.ContextComponent], # component class name: component instance
            component_order: Sequence[str] | None = None,
            pre_act_key: str = DEFAULT_PRE_ACT_KEY,
            logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
            social_personality: str = 'no_social_personality',
            flexible_action_parsing: bool = False,
            # svo_component: ValueTracker | None = None,
    ):

      self._model = model
      self._clock = clock
      self.social_personality = social_personality
      self._flexible_action_parsing = flexible_action_parsing
      if component_order is None:
          self._component_order = None
      else:
          self._component_order = tuple(component_order)
      if self._component_order is not None:
          if len(set(self._component_order)) != len(self._component_order):
              raise ValueError(
                  'The component order contains duplicate components: '
                  + ', '.join(self._component_order)
              )

      self._pre_act_key = pre_act_key
      self._logging_channel = logging_channel
      self._num_proposed_actions = num_proposed_actions
      self._desire_component_dict = desire_component_dict
      self._desire_component_names = tuple(_get_class_name(compo) for compo in self._desire_component_dict.values())
      self._desire_name = tuple(self._desire_component_dict.keys())
      # Compact, read-only-after-write records for Replay export. Keeping these
      # records does not add model calls or participate in action selection.
      self._decision_history = []

    def svo_instruction(self) -> str:
        """"不同人格对应的SVO的说明"""
        total_prompt = ""
        if self.social_personality == "Altruistic":
            instruction_prompt = ("Altruistic: Select actions that bring the greatest benefit to others, "
                                  "even if it means sacrificing your own interests. Prioritize helping others above all else.\n")
        elif self.social_personality == "Prosocial":
            instruction_prompt = ("Choose actions that promote fairness and cooperation. "
                                  "Aim to balance both your own outcomes and others', seeking win-win solutions and minimizing inequality.\n")
        elif self.social_personality == "Individualistic":
            instruction_prompt = ("Individualistic: Pick actions that maximize your own gain, regardless of the consequences for others. "
                                  "Focus only on your personal success and advantage.\n")
        elif self.social_personality == "Competitive":
            instruction_prompt = ("Competitive**: Choose actions that increase your advantage over others, even if your absolute outcome is not optimal. "
                                  "Prioritize outperforming others, sometimes at the expense of collective benefit.\n")
        else:
            raise ValueError(f'social_personality \'{self.social_personality}\' is not in SVO_RANGES')
        total_prompt += instruction_prompt
        # total_prompt += expected_svo_prompt
        # total_prompt += RANGES_prompt
        return total_prompt

    def _svo_recovery_prompt(self) -> str:
        """Return a one-sentence gentle nudge if current social-benefit balance
        is outside the expected band for this personality type."""
        try:
            svo_value = (
                self.get_entity()
                .get_component("ValueTracker", type_=desire_svo_comp.ValueTracker)
                .get_current_svo_value()
            )
        except AttributeError:
            return ""  # ValueTracker 不存在时静默
        agent_name = self.get_entity().name
        low, high = SVO_RANGES_QUARTILE[self.social_personality]
        if low <= svo_value <= high:
            return ""

        # ── 不显式出现 SVO / 回正 ──
        if svo_value < low:
            # 行为过于自利 → 提醒更多顾及他人
            hint = f"{agent_name} should also consider his/her own gains when comparing the options when take actions. "
        else:  # svo_value > high
            # 行为过于利他 → 提醒关注自身
            hint = f"{agent_name} should give extra weight to actions that raise others' satisfaction when take actions. "

        return (
            f"{agent_name}\'s current SVO value deviates from the expected value, "
            f"{hint}\n"
        )

    def _get_desire_status(self):
        desire_status = ''
        agent_name = self.get_entity().name
        for name, compo in self._desire_component_dict.items():
          desire_status += f"{agent_name}'s {name}: {compo.get_pre_act_value()}\n"
        return desire_status
    #_get_desire_status的作用是获取所有的 desire 的状态

    def _get_desire_name(self):
        return self._desire_name
    #_get_desire_name的作用是获取所有的 desire 的名字，并储存在 _desire_name 中（元组）

    def _context_for_action(
        self,
        contexts: entity_component.ComponentContextMapping,
    ) -> str:
        # _context_for_action 的作用是将所有的 context 进行拼接
        if self._component_order is None:
            return '\n'.join(
                context for context in contexts.values() if context
        )

        desire_set = set(self._desire_component_names)
        filtered_component_order = tuple(item for item in self._component_order if item not in desire_set)
        filtered_context_keys = tuple(sorted(set(contexts.keys()) - set(self._component_order) - desire_set))

        order = filtered_component_order + filtered_context_keys + self._desire_component_names

        desire_context = '\n'.join(contexts[name] for name in order if contexts[name])
        return desire_context

    def _preprocess_imagined_action(self, imagined_actions: str) -> list:
        """Extract numbered activities, or one concrete unnumbered action."""
        if not self._flexible_action_parsing:
            actions = []
            for index in range(self._num_proposed_actions):
                marker = f'Activity {index + 1}: '
                if marker not in imagined_actions:
                    break
                actions.append(imagined_actions.split(marker, 1)[1].split('\n', 1)[0])
            return actions
        answer = str(imagined_actions).strip()
        markers = list(re.finditer(
            r'(?im)^\s*(?:(?:Activity|Action)\s*)?([1-9])\s*[:.)-]\s*', answer
        ))
        actions = []
        for index, marker in enumerate(markers):
            if len(actions) >= self._num_proposed_actions:
                break
            end = markers[index + 1].start() if index + 1 < len(markers) else len(answer)
            candidate = answer[marker.end():end].strip().splitlines()[0].strip()
            if candidate:
                actions.append(candidate)
        if actions:
            return actions
        answer = re.sub(r'^Answer\s*:\s*', '', answer, flags=re.IGNORECASE)
        candidate = answer.splitlines()[0].strip() if answer else ''
        if candidate and not re.match(r'(?i)^(?:no feasible actions|cannot provide|unable to)', candidate):
            return [candidate]
        return []

    @staticmethod
    def _normalize_desire_key(value: str) -> str:
        return ''.join(character for character in value.lower() if character.isalnum())

    def _parse_predicted_desires(self, predicted_states: str) -> dict:
        """Extract only explicitly emitted desire-state lines from the LLM output."""
        known_desires = {
            self._normalize_desire_key(name): self._normalize_desire_key(name)
            for name in self._desire_name
        }
        parsed = {}
        for line in str(predicted_states).splitlines():
            if ':' not in line:
                continue
            raw_name, raw_state = line.split(':', 1)
            key = known_desires.get(self._normalize_desire_key(raw_name))
            state = raw_state.strip()
            if key and state:
                parsed[key] = state
        return parsed

    def get_decision_history(self) -> list[dict]:
        """Return action-decision records already generated by ASVO."""
        return list(self._decision_history)

    def _imagine_result_of_action(self, proposed_action):
        """"对 proposed_action 进行想象，返回一个字典，包含状态和提示"""
        prompt = interactive_document.InteractiveDocument(self._model)

        agent_name = self.get_entity().name
        imagine_prompt = (f'{agent_name} is a human-like agent, '
                          f'{agent_name} will receive a series of observations describing desires in many dimensions '
                          'and an action taken in the current time step. '
                          f'{agent_name} needs to first analyze how desires change after the action taken, '
                          'and then output the states of desire-state observations in the same format as the input.\n')

        action_context = f'{agent_name} takes the action: \n{proposed_action} \n'
        desire_status = f"{agent_name}'s original desire states: \n{self._get_desire_status()}\n"
        output_format = (f"Please output the states of desire-state observations in the following format: \n")
        for desire in self._desire_name:
            output_format += f"{desire}: <{desire} state> \n"
        total_prompt = imagine_prompt + desire_status + action_context + output_format
        imagined_states = prompt.open_question(
            total_prompt,
            max_tokens=2200,
            terminators=(),
            question_label='Exercise',
        )
        return {"status": imagined_states, "prompt": prompt.view().text()}
    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        prompt = interactive_document.InteractiveDocument(self._model)
        context = self._context_for_action(contexts)#将所有的 context 进行拼接
        prompt.statement(context + '\n')
        agent_name = self.get_entity().name#get_entity()返回对应的实体
        # agent_personality = self.social_personality_check()
        agent_personality = self.social_personality
        MCTS_log = dict()
        MCTS_log['component context'] = context

        tree_thinking_prompt = (f"{agent_name} is a human-like agent. "
                                f"and {agent_name} has a value system of {agent_personality}. "
                                )
        tree_thinking_prompt+=  self.svo_instruction()
        svo_recovery_prompt = self._svo_recovery_prompt()
        tree_thinking_prompt += svo_recovery_prompt
        tree_thinking_prompt += (
                        f"{agent_name}\'s behavior should consistently align with your assigned Social Value Orientation (SVO) type. "
                        f"{agent_name} will observe current states over "
                        f"{', '.join(self._desire_name)} "
                        f"which represent {len(self._desire_name)} desire or value dimensions. "
                        "Given these state descriptions, "
                        f"please generate {self._num_proposed_actions} activities "
                        "(which may contain several feasible actions for each) "
                        f"that might have the most positive impact on {agent_name}'s desires (including physical and mental)."
                        f"{agent_name} needs to focus on immediate desires and take activities that can satisfy them, "
                        f"but at the same time, {agent_name} must ensure that actions are reasonable and varied. "
                        f"Notice that {agent_name} can only interact with items provided by the environment. "
                        f"{agent_name} needs to describe activities in a more specific mode and ensure "
                        "that the time required for the action sequence matches the required time period. "
                        f"Please output the {self._num_proposed_actions} activities in the following format:\n"
                        "'Activity 1: <first possible action sequence> \n"
                        "Activity 2: <second possible action sequence> \n"
                        "Activity 3: <third possible action sequence> \n ......' "
                        "and ensure that the time required for the action sequence matches the required time period.")
        call_to_action = action_spec.call_to_action.format(
            name=self.get_entity().name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
        )

        tree_thinking_prompt = tree_thinking_prompt + '\n' + call_to_action
        tree_thinking_answer = prompt.open_question(
            tree_thinking_prompt,
            max_tokens=1200,
            terminators=(),
            question_label='Exercise',
        )
        print(prompt.view().text())
        MCTS_log['tree_thinking_prompt'] = prompt.view().text()
        MCTS_log['tree_thinking_answer'] = tree_thinking_answer
        imagined_actions = self._preprocess_imagined_action(tree_thinking_answer)
        MCTS_log['imagined_actions'] = imagined_actions

        # 对所有可能的动作进行想象，返回一个字典，包含行动之后的desire状态和提示
        result_of_imagined_actions = concurrency.run_tasks({
          query: functools.partial(self._imagine_result_of_action, query)
          for query in imagined_actions
        })

        MCTS_log['result_of_imagined_actions'] = result_of_imagined_actions
        result_of_imagined_actions = {key: value['status'] for key, value in result_of_imagined_actions.items()}

        prompt = interactive_document.InteractiveDocument(self._model)

        desire_status = self._get_desire_status()
        observation_status = self.get_entity().get_component("Observation", type_=agent_components.action_spec_ignored.ActionSpecIgnored).get_pre_act_value()
        action_selection_prompt = (
                           f"{agent_name} is a human-like agent. "
                           f"{agent_name} will first receive a series of observations "
                           "describing the current states of desires in many dimensions. "
                           f"Then, {agent_name} will receive several feasible actions along with the states of desire "
                           "after taking each action. "
                           f"{agent_name} needs to compare these actions and their corresponding states of desire, "
                           f"and choose the action that has the most positive impact on all desires of {agent_name} "
                           f"or value states. {agent_name} should focus on immediate desires "
                           "and take actions that can satisfy them. "
                           )
        action_selection_svo_instruction = self.svo_instruction()
        action_selection_prompt += action_selection_svo_instruction
        action_selection_prompt += (
                            f"The observations of the surrounding environment: \n"
                            f"{observation_status} \n"
                            f"{agent_name}'s current states of desire: \n"
                            f"{desire_status} \n"
        )
        action_and_result = f"Following are the states of desire after each action: \n"
        for i in range(self._num_proposed_actions):
          if i >= len(imagined_actions):
            break
          action = imagined_actions[i]
          action_and_result += f'Action {i+1}: {action} \n'
          action_and_result += f'States after action {i+1}: {result_of_imagined_actions[action]} \n\n'

        action_selection_prompt += action_and_result

        action_selection_prompt += (f'Please output the specific best action instead without explanation of <Action 1> or <Action 2> and so on. '
                                    'If there is only one action provided, output the action content directly. \n')
        action_selection_prompt += (f"Please output the best action in the following format: \n"
                                    f"'Action: <{agent_name}'s best action>' \n"
                                    f"Example: Action: {agent_name} observes the surroundings.\n")

        o = prompt.open_question(action_selection_prompt, max_tokens=2200,terminators=())

        if o.startswith('Action'):
            o = o.split('Action', 1)[1].strip(' :')
        output = o
        MCTS_log['action_selection_prompt'] = prompt.view().text()
        MCTS_log['action_selection_answer'] = output

        value_tracker = self.get_entity().get_component(
            "ValueTracker", type_=desire_svo_comp.ValueTracker
        )
        current_desires = {
            self._normalize_desire_key(name): component.get_current_numerical_value_without_update()
            for name, component in self._desire_component_dict.items()
        }
        candidates = []
        for candidate_action in imagined_actions:
            prediction = result_of_imagined_actions.get(candidate_action)
            candidates.append({
                'action': candidate_action,
                # This implementation predicts desire-state text, not a
                # separate environmental outcome or numeric desire vector.
                'predicted_outcome': None,
                'predicted_desires': self._parse_predicted_desires(prediction),
                'prediction_raw': prediction,
                'prediction_format': 'qualitative_text',
                'reason': None,
                'utility': None,
                'self_satisfaction': None,
                'svo_alignment': None,
                'resulting_svo': None,
                'selected': output.strip() == candidate_action.strip(),
            })
        self._decision_history.append({
            'selected_action': output,
            'selected_candidate_index': next(
                (index for index, candidate in enumerate(candidates)
                 if candidate['selected']),
                None,
            ),
            # The selection prompt explicitly asks for no explanation.
            'selection_reason': None,
            'observation': observation_status,
            # Exact qualitative desire-state text supplied to selection.
            'current_desire_states_raw': desire_status,
            'current_desires': current_desires,
            'current_svo': float(value_tracker.get_current_svo_value()),
            'configured_social_personality': self.social_personality,
            'svo_instruction': action_selection_svo_instruction,
            'svo_regulation_prompt': svo_recovery_prompt or None,
            'candidates': candidates,
        })
        self._log(MCTS_log, prompt)

        return output

    def _log(self,
           result: str,
           prompt: interactive_document.InteractiveDocument):
        self._logging_channel({
            'Key': self._pre_act_key,
            'Value': result,
            'Prompt': prompt.view().text().splitlines(),
        })
    #_log函数的作用是将结果和提示信息记录到日志中

    def get_state(self) -> entity_component.ComponentState:
        """Converts the component to a dictionary."""
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
      pass
