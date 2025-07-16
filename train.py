from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--distillation_loss', type=str, default='cwd')
    parser.add_argument('--mode', type=str, choices=['distill', 'teacher', 'student'], 
                        default='distill', help='Training mode: distill, teacher, or student')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load models based on training mode
    if args.mode in ['distill', 'student']:
        student_model = YOLO("yolo11n.pt")
    
    if args.mode in ['distill', 'teacher']:
        teacher_model = YOLO("yolo11l.pt")

    # Training based on mode
    if args.mode == 'distill':
        # Distillation training
        student_model.train(
            data="dataset_sliced/data.yaml",
            teacher=teacher_model.model,
            distillation_loss=args.distillation_loss,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
            optimizer='Adam',
            momentum=0.9,
            weight_decay=0.0001,
            lr0=0.001,
        )
    elif args.mode == 'teacher':
        # Train teacher model only
        teacher_model.train(
            data="dataset_sliced/data.yaml",
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
        )
    elif args.mode == 'student':
        student_model.train(
            data="dataset_sliced/data.yaml",
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
            optimizer='Adam',
            momentum=0.9,
            weight_decay=0.0001,
            lr0=0.001,
        )

if __name__ == "__main__":
    main()