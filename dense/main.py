from head_mimo import *




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = get_config()
    log(args)

    # model
    model = mmsenet(args)

    # main
    if args.phase == 'train':
        train = Trainer(args, model)
        train.tr()

    elif args.phase == 'test':
        test = Tester(args, model)
        test.test()

    print('[*] Finish!')