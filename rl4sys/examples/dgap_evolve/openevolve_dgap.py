import argparse
from rl4sys.examples.dgap_evolve.dgap_evaluator import DGAPEvaluator


def main():
  parser = argparse.ArgumentParser(description='OpenEvolve DGAP Runner')
  parser.add_argument('--nv', default=88581, type=int, help='Number of vertices')
  parser.add_argument('--ne', default=37598, type=int, help='Number of edges')
  parser.add_argument('--base_file', default='rl4sys/examples/dgap/sx-mathoverflow-unique-undir.base.el', type=str)
  parser.add_argument('--dynamic_file', default='rl4sys/examples/dgap/sx-mathoverflow-unique-undir.dynamic.el', type=str)
  parser.add_argument('--rl_mode', default='continuous', choices=['continuous', 'discrete'])
  parser.add_argument('--repeats', default=1, type=int)
  parser.add_argument('--tag', default='openevolve', type=str)
  args = parser.parse_args()

  evaluator = DGAPEvaluator(args.base_file, args.dynamic_file, args.nv, args.ne)
  summary = evaluator.compare(rl_mode=args.rl_mode, tag_prefix=args.tag, repeats=args.repeats)
  print('Evaluation Summary:')
  print(summary)


if __name__ == '__main__':
  main()


